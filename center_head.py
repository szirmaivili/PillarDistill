import logging
import copy
import tensorflow as tf
from collections import defaultdict

import box_tf_ops
from kaiming import kaiming_init_tf
from centernet_loss import FastFocalLoss, RegLoss, IouLoss, IouRegLoss
from center_utils import circle_nms

class SepHead(tf.keras.Model):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        self.sub_heads = {}

        for head, (classes, num_conv) in self.heads.items():
            layers = []

            for i in range(num_conv - 1):
                layers.append(tf.keras.layers.Conv2D(
                    filters=head_conv,
                    kernel_size=final_kernel,
                    strides=1,
                    padding="same",
                    use_bias=True
                ))
                if bn:
                    layers.append(tf.keras.layers.BatchNormalization())

                layers.append(tf.keras.layers.ReLU(max_value=6)) # Ezt kérdezd meg, hogy hogy kell beállítani

            layers.append(tf.keras.layers.Conv2D(
                filters=classes,
                kernel_size=final_kernel,
                strides=1,
                padding="same",
                use_bias=True
            ))

            head_net = tf.keras.Sequential(layers, name=head)

            # inicializáció
            if 'hm' in head:
                layers.append(tf.keras.layers.Conv2D(
                    filters=classes,
                    kernel_size=final_kernel,
                    strides=1,
                    padding="same",
                    use_bias=True,
                    bias_initializer=tf.constant_initializer(init_bias)  # <<< EZ ITT
            ))
            else:
                layers.append(tf.keras.layers.Conv2D(
                    filters=classes,
                    kernel_size=final_kernel,
                    strides=1,
                    padding="same",
                    use_bias=True
                ))


            self.sub_heads[head] = head_net

    def call(self, x, training=True):
        ret_dict = {}
        for head, net in self.sub_heads.items():
            ret_dict[head] = net(x, training=training)
        return ret_dict

class CenterHead(tf.keras.Model):
    def __init__(
        self,
        in_channels=[128],
        tasks=[],
        dataset='nuscenes',
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
        iou_reg=None,
        **kwargs
    ):
        super(CenterHead, self).__init__(**kwargs)

        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.task_strides = [str(t["stride"]) for t in tasks]
        self.code_weights = code_weights
        self.weight = weight
        self.dataset = dataset

        import itertools
        order_class_names = list(itertools.chain(*self.class_names))

        self.class_id_mapping_each_head = []
        for cur_class_names in self.class_names:
            cur_class_id_mapping = tf.constant(
                [order_class_names.index(x) for x in cur_class_names],
                dtype=tf.int64
            )
            self.class_id_mapping_each_head.append(cur_class_id_mapping)


        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger
        logger.info(f"num_classes: {self.num_classes}")

        # közös konvolúció
        self.shared_conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(share_conv_channel, 3, padding="same", use_bias=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

        # task fejek
        self.tasks = []
        for num_cls in self.num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, bn=True,
                            init_bias=init_bias, final_kernel=3)
                )
            else:
                raise NotImplementedError("DCN head not ported to TF")

        logger.info("Finish CenterHead Initialization")

        # loss függvények
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        self.with_iou = 'iou' in common_heads
        self.with_iou_reg = iou_reg is not None
        if self.with_iou:
            self.crit_iou = IouLoss()
        self.crit_iou_reg = None
        if self.with_iou_reg:
            self.crit_iou_reg = IouRegLoss(iou_reg)

        self.box_n_dim = 9 if 'vel' in common_heads else 7
        self.use_direction_classifier = False

    def call(self, x, training=False):
        ret_dicts = []
        x = self.shared_conv(x, training=training)
        for task in self.tasks:
            ret_dicts.append(task(x, training=training))
        return ret_dicts, x

    def _sigmoid(self, x):
        return tf.clip_by_value(tf.sigmoid(x), 1e-4, 1 - 1e-4)

    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(
                preds_dict['hm'],
                example['hm'][task_id],
                example['ind'][task_id],
                example['mask'][task_id],
                example['cat'][task_id]
            )

            target_box = example['anno_box'][task_id]
            if self.dataset in ['waymo', 'nuscenes']:
                if 'vel' in preds_dict:
                    preds_dict['anno_box'] = tf.concat(
                        [preds_dict['reg'], preds_dict['height'],
                         preds_dict['dim'], preds_dict['vel'],
                         preds_dict['rot']], axis=-1)
                else:
                    preds_dict['anno_box'] = tf.concat(
                        [preds_dict['reg'], preds_dict['height'],
                         preds_dict['dim'], preds_dict['rot']], axis=-1)
                    target_box = tf.gather(target_box, [0, 1, 2, 3, 4, 5, -2, -1], axis=-1)
            else:
                raise NotImplementedError()

            ret = {}

            box_loss = self.crit_reg(
                preds_dict['anno_box'],
                example['mask'][task_id],
                example['ind'][task_id],
                target_box
            )

            loc_loss = tf.reduce_sum(box_loss * tf.constant(self.code_weights, dtype=box_loss.dtype))

            loss = hm_loss + self.weight * loc_loss
            ret.update({
                'hm_loss': hm_loss,
                'loc_loss': loc_loss,
                'loc_loss_elem': box_loss,
                'num_positive': tf.reduce_sum(tf.cast(example['mask'][task_id], tf.float32))
            })

            if self.with_iou or self.with_iou_reg:
                batch_dim = tf.exp(tf.clip_by_value(preds_dict['dim'], -5, 5))
                #batch_dim = tf.transpose(batch_dim, [0, 2, 3, 1])
                batch_rot = preds_dict['rot'] #tf.transpose(preds_dict['rot'], [0, 2, 3, 1])
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                # batch_reg = tf.transpose(preds_dict['reg'], [0, 2, 3, 1])
                # batch_hei = tf.transpose(preds_dict['height'], [0, 2, 3, 1])
                batch_rot = tf.atan2(batch_rots, batch_rotc)

                batch, H, W, _ = tf.unstack(tf.shape(batch_dim))

                batch_reg = tf.reshape(preds_dict['reg'], [batch, H * W, 2])
                batch_hei = tf.reshape(preds_dict['height'], [batch, H * W, 1])
                batch_rot = tf.reshape(batch_rot, [batch, H * W, 1])
                batch_dim = tf.reshape(batch_dim, [batch, H * W, 3])

                ys, xs = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
                ys = tf.cast(ys[None, ...], batch_dim.dtype)
                xs = tf.cast(xs[None, ...], batch_dim.dtype)

                xs = tf.reshape(xs, [1, -1, 1]) + batch_reg[:, :, 0:1]
                ys = tf.reshape(ys, [1, -1, 1]) + batch_reg[:, :, 1:2]

                xs = xs * int(self.task_strides[task_id]) * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
                ys = ys * int(self.task_strides[task_id]) * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
                batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_rot], axis=2)
                batch_box_preds = tf.transpose(batch_box_preds, [0, 2, 1])
                batch_box_preds = tf.reshape(batch_box_preds, [batch, -1, H, W])

                if self.with_iou:
                    pred_boxes_for_iou = tf.stop_gradient(batch_box_preds)
                    iou_loss = self.crit_iou(
                        preds_dict['iou'], example['mask'][task_id], example['ind'][task_id],
                        pred_boxes_for_iou, example['gt_box'][task_id]
                    )
                    loss = loss + iou_loss
                    ret.update({'iou_loss': iou_loss})

                if self.with_iou_reg:
                    iou_reg_loss = self.crit_iou_reg(
                        batch_box_preds,
                        example['mask'][task_id],
                        example['ind'][task_id],
                        example['gt_box'][task_id]
                    )
                    loss = loss + self.weight * iou_reg_loss
                    ret.update({'iou_reg_loss': iou_reg_loss})

            ret.update({'loss': loss})
            rets.append(ret)

        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    # @tf.function
    # def predict(self, example, preds_dicts, test_cfg, **kwargs):
    #     """decode + NMS"""
    #     rets = []
    #     metas = []

    #     post_center_range = tf.constant(
    #         test_cfg.post_center_limit_range,
    #         dtype=preds_dicts[0]['hm'].dtype
    #     )

    #     for task_id, preds_dict in enumerate(preds_dicts):
    #         batch_size = preds_dict['hm'].shape[0]

    #         batch_hm = tf.sigmoid(preds_dict['hm'])
    #         batch_dim = tf.exp(tf.clip_by_value(preds_dict['dim'], -5, 5))

    #         if 'iou' in preds_dict:
    #             batch_iou = (preds_dict['iou'][..., 0] + 1) * 0.5
    #         else:
    #             batch_iou = tf.ones(batch_hm.shape[:-1], dtype=batch_dim.dtype)

    #         batch_rots = preds_dict['rot'][..., 0:1]
    #         batch_rotc = preds_dict['rot'][..., 1:2]
    #         batch_reg = preds_dict['reg']
    #         batch_hei = preds_dict['height']
    #         batch_rot = tf.atan2(batch_rots, batch_rotc)

    #         batch, H, W, num_cls = batch_hm.shape

    #         batch_reg = tf.reshape(batch_reg, [batch, H * W, 2])
    #         batch_hei = tf.reshape(batch_hei, [batch, H * W, 1])
    #         batch_rot = tf.reshape(batch_rot, [batch, H * W, 1])
    #         batch_dim = tf.reshape(batch_dim, [batch, H * W, 3])
    #         batch_hm = tf.reshape(batch_hm, [batch, H * W, num_cls])

    #         ys, xs = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
    #         ys = tf.cast(ys[None, ...], batch_dim.dtype)
    #         xs = tf.cast(xs[None, ...], batch_dim.dtype)

    #         xs = tf.reshape(xs, [1, -1, 1]) + batch_reg[:, :, 0:1]
    #         ys = tf.reshape(ys, [1, -1, 1]) + batch_reg[:, :, 1:2]

    #         xs = xs * int(self.task_strides[task_id]) * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
    #         ys = ys * int(self.task_strides[task_id]) * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

    #         batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_rot], axis=2)

    #         rets.append(self.post_processing(batch_box_preds, batch_hm, batch_iou,
    #                                          test_cfg, post_center_range, task_id))
    #         metas.append(example.get("metadata", [None] * batch_size))

    #     return rets

    @tf.function
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode + NMS + optional double flip testing"""
        rets = []
        metas = []

        double_flip = True

        post_center_range = tf.constant(
            test_cfg.post_center_limit_range,
            dtype=preds_dicts[0]['hm'].dtype
        )

        for task_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                tf.debugging.assert_equal(batch_size % 4, 0, "Batch size must be multiple of 4 for double flip")
                base_bs = batch_size // 4

                for k, v in preds_dict.items():
                    # shape: [B, H, W, C] → [B/4, 4, H, W, C]
                    _, H, W, C = v.shape
                    v = tf.reshape(v, [base_bs, 4, H, W, C])

                    # 2. index: y-flip
                    v_yflip = tf.reverse(v[:, 1], axis=[1])  # flip H dim
                    # 3. index: x-flip
                    v_xflip = tf.reverse(v[:, 2], axis=[2])  # flip W dim
                    # 4. index: xy-flip
                    v_xyflip = tf.reverse(v[:, 3], axis=[1, 2])

                    preds_dict[k] = tf.stack([v[:, 0], v_yflip, v_xflip, v_xyflip], axis=1)

            # heatmap, dim, iou
            batch_hm = tf.sigmoid(preds_dict['hm'])
            batch_dim = tf.exp(tf.clip_by_value(preds_dict['dim'], -5, 5))

            if 'iou' in preds_dict:
                batch_iou = (preds_dict['iou'][..., 0] + 1) * 0.5
            else:
                batch_iou = tf.ones(batch_hm.shape[:-1], dtype=batch_dim.dtype)

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                # Átlagolás az augm. 4 csoportján
                batch_hm = tf.reduce_mean(batch_hm, axis=1)
                batch_iou = tf.reduce_mean(batch_iou, axis=1)
                batch_hei = tf.reduce_mean(batch_hei, axis=1)
                batch_dim = tf.reduce_mean(batch_dim, axis=1)

                # Regresszió flip-korrekció
                batch_reg = tf.reduce_mean(batch_reg, axis=1)

                # Rotáció flip-korrekció
                # y-flip: cos vált előjelet
                rotc_yflip = batch_rotc[:, 1] * -1.0
                # x-flip: sin vált előjelet
                rots_xflip = batch_rots[:, 2] * -1.0
                # xy-flip: mindkettő vált
                rotc_xyflip = batch_rotc[:, 3] * -1.0
                rots_xyflip = batch_rots[:, 3] * -1.0

                # átlagoljuk őket
                batch_rotc = tf.reduce_mean(tf.stack([batch_rotc[:, 0],
                                                    rotc_yflip,
                                                    batch_rotc[:, 2],
                                                    rotc_xyflip], axis=1), axis=1)

                batch_rots = tf.reduce_mean(tf.stack([batch_rots[:, 0],
                                                    batch_rots[:, 1],
                                                    rots_xflip,
                                                    rots_xyflip], axis=1), axis=1)

            batch_rot = tf.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.shape

            batch_reg = tf.reshape(batch_reg, [batch, H * W, 2])
            batch_hei = tf.reshape(batch_hei, [batch, H * W, 1])
            batch_rot = tf.reshape(batch_rot, [batch, H * W, 1])
            batch_dim = tf.reshape(batch_dim, [batch, H * W, 3])
            batch_hm = tf.reshape(batch_hm, [batch, H * W, num_cls])

            ys, xs = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
            ys = tf.cast(ys[None, ...], batch_dim.dtype)
            xs = tf.cast(xs[None, ...], batch_dim.dtype)

            xs = tf.reshape(xs, [1, -1, 1]) + batch_reg[:, :, 0:1]
            ys = tf.reshape(ys, [1, -1, 1]) + batch_reg[:, :, 1:2]

            xs = xs * int(self.task_strides[task_id]) * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * int(self.task_strides[task_id]) * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']  # (B, 4, H, W, 2) double_flip esetén

                if double_flip:
                    # flip vy ([:,1,...,1])
                    vy_flipped = batch_vel[:, 1, ..., 1] * -1
                    vy_flipped = tf.expand_dims(vy_flipped, axis=-1)  # visszaalakítjuk a shape-et
                    vel1 = tf.concat([batch_vel[:, 1, ..., :1], vy_flipped], axis=-1)

                    # flip vx ([:,2,...,0])
                    vx_flipped = batch_vel[:, 2, ..., 0] * -1
                    vx_flipped = tf.expand_dims(vx_flipped, axis=-1)
                    vel2 = tf.concat([vx_flipped, batch_vel[:, 2, ..., 1:]], axis=-1)

                    # dupla flip ([:,3])
                    vel3 = batch_vel[:, 3] * -1

                    # összerakjuk vissza (0. index változatlan marad)
                    batch_vel = tf.stack(
                        [batch_vel[:, 0], vel1, vel2, vel3],
                        axis=1
                    )

                    # átlagolás a 4 variánson
                    batch_vel = tf.reduce_mean(batch_vel, axis=1)

                # végső reshape
                batch_vel = tf.reshape(batch_vel, [batch, H * W, 2])

                # box predikciókhoz hozzáadjuk
                batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], axis=2)
            else: 
                batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_rot], axis=2)

            rets.append(self.post_processing(batch_box_preds, batch_hm, batch_iou,
                                            test_cfg, post_center_range, task_id))
            metas.append(example.get("metadata", [None] * batch_size))

        num_samples = batch  # batch méret

        print('Num samples: ', num_samples)

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    # fűzd össze a többi taskból
                    ret[k] = tf.concat([task[i][k] for task in rets], axis=0)
                elif k == "label_preds":
                    # a class_id_mapping_each_head alapján mappelni kell
                    mapped = []
                    for j, cur_class_id_mapping in enumerate(self.class_id_mapping_each_head):
                        rets[j][i][k] = tf.gather(cur_class_id_mapping, rets[j][i][k])
                    ret[k] = tf.concat([ret[i][k] for ret in rets], axis=0)

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list


    def post_processing(self, batch_box_preds, batch_hm, batch_iou, test_cfg, post_center_range, task_id):
        batch_size = int(batch_hm.shape[0])
        prediction_dicts = []

        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            iou_preds = tf.reshape(batch_iou[i], [-1])
            num_class = hm_preds.shape[1]

            scores = tf.reduce_max(hm_preds, axis=-1)
            labels = tf.argmax(hm_preds, axis=-1, output_type=tf.int32)

            score_mask = scores > test_cfg.score_threshold
            dist_mask = tf.reduce_all(box_preds[:, :3] >= post_center_range[:3], axis=1) & \
                        tf.reduce_all(box_preds[:, :3] <= post_center_range[3:], axis=1)
            mask = score_mask & dist_mask

            box_preds = tf.boolean_mask(box_preds, mask)
            scores = tf.boolean_mask(scores, mask)
            labels = tf.boolean_mask(labels, mask)
            iou_preds = tf.clip_by_value(tf.boolean_mask(iou_preds, mask), 0.0, 1.0)

            boxes_for_nms = tf.gather(box_preds, [0, 1, 2, 3, 4, 5, 6], axis=1)

            if test_cfg.circular_nms:
                dets = tf.stack([boxes_for_nms[:, 0], boxes_for_nms[:, 1], scores], axis=-1)
                keep = circle_nms(dets.numpy(), thresh=test_cfg.min_radius[task_id])
                keep = keep[:test_cfg.nms.nms_post_max_size[task_id]]
                selected_boxes = tf.gather(box_preds, keep)
                selected_scores = tf.gather(scores, keep)
                selected_labels = tf.gather(labels, keep)
            else:
                selected = box_tf_ops.rotate_nms_pcdet(
                    boxes_for_nms, scores,
                    thresh=test_cfg.nms["nms_iou_threshold"],
                    pre_maxsize=test_cfg.nms["nms_pre_max_size"],
                    post_max_size=test_cfg.nms["nms_post_max_size"]
                )
                selected_boxes = tf.gather(box_preds, selected)
                selected_scores = tf.gather(scores, selected)
                selected_labels = tf.gather(labels, selected)

            prediction_dicts.append({
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            })

        return prediction_dicts
