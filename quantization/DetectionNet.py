import torch
import torch.nn as nn
import numpy as np

from backbone import Backbone
from neck import Neck
from bbox_heads import Bbox
from collections import defaultdict
from det3d.models.readers.dynamic_pillar_encoder import DynamicPillarFeatureNet
from det3d.core import box_torch_ops
from det3d.core.utils.circle_nms_jit import circle_nms
import itertools
import sys
import pickle
from datetime import datetime

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  

tasks = [
    dict(stride=8, class_names=["car"]),
    dict(stride=8, class_names=["truck", "construction_vehicle"]),
    dict(stride=8, class_names=["bus", "trailer"]),
    dict(stride=8, class_names=["barrier"]),
    dict(stride=8, class_names=["motorcycle", "bicycle"]),
    dict(stride=8, class_names=["pedestrian", "traffic_cone"]),
]

class DetectionNet(nn.Module):
    def __init__(self, 
                 backbone: Backbone, 
                 neck: Neck, 
                 bbox_head: Bbox,
                 reader: DynamicPillarFeatureNet,
                 crit,
                crit_reg,
                crit_iou=None,
                crit_iou_reg=None,
                code_weights=None,
                weight=0.25,
                with_iou=False,
                with_iou_reg=False,
                task_strides=None,
                dataset="nuscenes",
                train_cfg=None,
                test_cfg=None):
        
        super(DetectionNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.reader = reader

        self.crit = crit
        self.crit_reg = crit_reg
        self.crit_iou = crit_iou
        self.crit_iou_reg = crit_iou_reg
        self.code_weights = code_weights or [1.0] * 10
        self.weight = weight
        self.with_iou = with_iou
        self.with_iou_reg = with_iou_reg
        self.task_strides = task_strides or [8] * 6
        self.dataset = dataset
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]

        order_class_names = list(itertools.chain(*self.class_names))

        self.class_id_mapping_each_head = []
        for cur_class_names in self.class_names:
            cur_class_id_mapping = torch.tensor(
                [order_class_names.index(x) for x in cur_class_names],
                dtype=torch.int64).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

    def forward(self, example, return_loss=True):
        # Backbone → két feature map (conv_4, conv_5)

        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )

        sp_tensor = self.reader(data)
        conv_4, conv_5 = self.backbone(sp_tensor)
        
        # Neck → összefűzi és feldolgozza a feature mapeket
        neck_out = self.neck(conv_4, conv_5)

        output = self.bbox_head(neck_out)

        if return_loss:

            return self.loss(example, output, self.test_cfg)
        
        else:

            return self.predict(example, output, self.test_cfg)

    def loss(self, example, preds_dicts, test_cfg):

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            preds_dict['hm'] = torch.sigmoid(preds_dict['hm'])

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
                    preds_dict['anno_box'] = torch.cat((
                        preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                        preds_dict['vel'], preds_dict['rot']
                    ), dim=1)
                else:
                    preds_dict['anno_box'] = torch.cat((
                        preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                        preds_dict['rot']
                    ), dim=1)
                    target_box = target_box[..., [0, 1, 2, 3, 4, 5, -2, -1]]
            else:
                raise NotImplementedError()

            box_loss = self.crit_reg(
                preds_dict['anno_box'],
                example['mask'][task_id],
                example['ind'][task_id],
                target_box
            )

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            loss = hm_loss + self.weight * loc_loss

            ret = {
                'hm_loss': hm_loss.detach().cpu(),
                'loc_loss': loc_loss,
                'loc_loss_elem': box_loss.detach().cpu(),
                'num_positive': example['mask'][task_id].float().sum()
            }

            # IoU komponensek
            if self.with_iou or self.with_iou_reg:
                batch_dim = torch.exp(torch.clamp(preds_dict['dim'], min=-5, max=5))
                batch_dim = batch_dim.permute(0, 2, 3, 1).contiguous()
                batch_rot = preds_dict['rot'].permute(0, 2, 3, 1).contiguous()
                batch_rots, batch_rotc = batch_rot[..., 0:1], batch_rot[..., 1:2]
                batch_reg = preds_dict['reg'].permute(0, 2, 3, 1).contiguous()
                batch_hei = preds_dict['height'].permute(0, 2, 3, 1).contiguous()
                batch_rot = torch.atan2(batch_rots, batch_rotc)

                B, H, W, _ = batch_dim.size()
                batch_reg = batch_reg.reshape(B, H * W, 2)
                batch_hei = batch_hei.reshape(B, H * W, 1)
                batch_rot = batch_rot.reshape(B, H * W, 1)
                batch_dim = batch_dim.reshape(B, H * W, 3)

                ys, xs = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
                ys = ys.view(1, H, W).repeat(B, 1, 1).to(batch_dim)
                xs = xs.view(1, H, W).repeat(B, 1, 1).to(batch_dim)
                xs = xs.view(B, -1, 1) + batch_reg[:, :, 0:1]
                ys = ys.view(B, -1, 1) + batch_reg[:, :, 1:2]

                xs = xs * int(self.task_strides[task_id]) * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
                ys = ys * int(self.task_strides[task_id]) * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)
                batch_box_preds = batch_box_preds.permute(0, 2, 1).contiguous().reshape(B, -1, H, W)

                if self.with_iou:
                    pred_boxes_for_iou = batch_box_preds.detach()
                    iou_loss = self.crit_iou(
                        preds_dict['iou'],
                        example['mask'][task_id],
                        example['ind'][task_id],
                        pred_boxes_for_iou,
                        example['gt_box'][task_id]
                    )
                    loss = loss + iou_loss
                    ret['iou_loss'] = iou_loss.detach().cpu()

                if self.with_iou_reg:
                    iou_reg_loss = self.crit_iou_reg(
                        batch_box_preds,
                        example['mask'][task_id],
                        example['ind'][task_id],
                        example['gt_box'][task_id]
                    )
                    loss = loss + self.weight * iou_reg_loss
                    ret['iou_reg_loss'] = iou_reg_loss.detach().cpu()

            ret['loss'] = loss
            rets.append(ret)

        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        return rets_merged
    
    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.double_flip

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):

            # convert N C H W to N H W C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 1).contiguous()

            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                #assert batch_size % 4 == 0, print(batch_size)

                if batch_size % 4 != 0:

                    print(f"[WARNING] Invalid batch_size={batch_size}, expected multiple of 4. Injecting dummy prediction.")

                    corrected_preds = {}
                    for k, v in preds_dict.items():
                        # v shape: (N, H, W, C)
                        if v.shape[0] < 4:
                            pad = 4 - v.shape[0]
                            pad_tensor = torch.zeros((pad, *v.shape[1:]), device=v.device, dtype=v.dtype)
                            corrected_preds[k] = torch.cat([v, pad_tensor], dim=0)
                        else:
                            corrected_preds[k] = v
                    preds_dict = corrected_preds

                batch_size = preds_dict['hm'].shape[0]

                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                if double_flip:
                    meta_list = meta_list[:4*int(batch_size):4]

            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_dim = torch.exp(torch.clamp(preds_dict['dim'].clone(), min=-5, max=5))
            if 'iou' in preds_dict.keys():
                batch_iou = (preds_dict['iou'].squeeze(dim=-1) + 1) * 0.5
                batch_iou = batch_iou.type_as(batch_dim)
            else:
                batch_iou = torch.ones((batch_hm.shape[0], batch_hm.shape[1], batch_hm.shape[2]),
                                        dtype=batch_dim.dtype).to(batch_hm.device)

            batch_rots = preds_dict['rot'][..., 0:1]
            batch_rotc = preds_dict['rot'][..., 1:2]
            batch_reg = preds_dict['reg']
            batch_hei = preds_dict['height']

            if double_flip:
                batch_hm = batch_hm.mean(dim=1)
                batch_iou = batch_iou.mean(dim=1)
                batch_hei = batch_hei.mean(dim=1)
                batch_dim = batch_dim.mean(dim=1)

                # y = -y reg_y = 1-reg_y
                batch_reg[:, 1, ..., 1] = 1 - batch_reg[:, 1, ..., 1]
                batch_reg[:, 2, ..., 0] = 1 - batch_reg[:, 2, ..., 0]

                batch_reg[:, 3, ..., 0] = 1 - batch_reg[:, 3, ..., 0]
                batch_reg[:, 3, ..., 1] = 1 - batch_reg[:, 3, ..., 1]
                batch_reg = batch_reg.mean(dim=1)

                # first yflip 
                # y = -y theta = pi -theta
                # sin(pi-theta) = sin(theta) cos(pi-theta) = -cos(theta)
                # batch_rots[:, 1] the same
                batch_rotc[:, 1] *= -1

                # then xflip x = -x theta = 2pi - theta
                # sin(2pi - theta) = -sin(theta) cos(2pi - theta) = cos(theta)
                # batch_rots[:, 2] the same
                batch_rots[:, 2] *= -1

                # double flip 
                batch_rots[:, 3] *= -1
                batch_rotc[:, 3] *= -1

                batch_rotc = batch_rotc.mean(dim=1)
                batch_rots = batch_rots.mean(dim=1)

            batch_rot = torch.atan2(batch_rots, batch_rotc)

            batch, H, W, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W, 2)
            batch_hei = batch_hei.reshape(batch, H*W, 1)

            batch_rot = batch_rot.reshape(batch, H*W, 1)
            batch_dim = batch_dim.reshape(batch, H*W, 3)
            batch_hm = batch_hm.reshape(batch, H*W, num_cls)

            ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])
            ys = ys.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W).repeat(batch, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

            xs = xs * int(self.task_strides[task_id]) * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * int(self.task_strides[task_id]) * test_cfg.voxel_size[1] + test_cfg.pc_range[1]

            if 'vel' in preds_dict:
                batch_vel = preds_dict['vel']

                if double_flip:
                    # flip vy
                    batch_vel[:, 1, ..., 1] *= -1
                    # flip vx
                    batch_vel[:, 2, ..., 0] *= -1

                    batch_vel[:, 3] *= -1
                    
                    batch_vel = batch_vel.mean(dim=1)

                batch_vel = batch_vel.reshape(batch, H*W, 2)
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_vel, batch_rot], dim=2)
            else: 
                batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, batch_rot], dim=2)

            metas.append(meta_list)

            if test_cfg.per_class_nms:
                pass 
            else:
                rets.append(self.post_processing(batch_box_preds, batch_hm, batch_iou, test_cfg, post_center_range, task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k in ["label_preds"]:
                    for j, cur_class_id_mapping in enumerate(self.class_id_mapping_each_head):
                        rets[j][i][k] = cur_class_id_mapping[rets[j][i][k]]
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

            ret['metadata'] = metas[0][i]
            ret_list.append(ret)

        return ret_list 

    @torch.no_grad()
    def post_processing(self, batch_box_preds, batch_hm, batch_iou, test_cfg, post_center_range, task_id):
        batch_size = len(batch_hm)

        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_preds[i]
            hm_preds = batch_hm[i]
            iou_preds = batch_iou[i].view(-1)
            num_class = hm_preds.shape[1]
            scores, labels = torch.max(hm_preds, dim=-1)

            print(torch.max(scores))

            score_mask = scores > test_cfg.score_threshold
            distance_mask = (box_preds[..., :3] >= post_center_range[:3]).all(1) \
                & (box_preds[..., :3] <= post_center_range[3:]).all(1)

            mask = distance_mask & score_mask 

            box_preds = box_preds[mask]
            scores = scores[mask]
            labels = labels[mask]
            iou_preds = torch.clamp(iou_preds[mask], min=0, max=1.)

            boxes_for_nms = box_preds[:, [0, 1, 2, 3, 4, 5, -1]]

            if test_cfg.circular_nms:
                centers = boxes_for_nms[:, [0, 1]] 
                boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                selected = _circle_nms(boxes, min_radius=test_cfg.min_radius[task_id],
                                       post_max_size=test_cfg.nms.nms_post_max_size[task_id])

                selected_boxes = box_preds[selected]
                selected_scores = scores[selected]
                selected_labels = labels[selected]
            elif test_cfg.nms.use_rotate_nms:
                scores = torch.pow(scores, 1-test_cfg.rectifier[task_id]) * torch.pow(iou_preds, test_cfg.rectifier[task_id])
                selected = box_torch_ops.rotate_nms_pcdet(boxes_for_nms.float(), scores.float(),
                                                          thresh=test_cfg.nms.nms_iou_threshold[task_id],
                                                          pre_maxsize=test_cfg.nms.nms_pre_max_size[task_id],
                                                          post_max_size=test_cfg.nms.nms_post_max_size[task_id])
                selected_boxes = box_preds[selected]
                selected_scores = scores[selected]
                selected_labels = labels[selected]
            elif test_cfg.nms.use_multi_class_nms:
                selected_boxes, selected_scores, selected_labels = box_torch_ops.rotate_class_specific_nms_iou_pcdet(
                    boxes_for_nms.float(), scores.float(), iou_preds, box_preds, labels, num_class,
                    test_cfg.rectifier[task_id],
                    thresh=test_cfg.nms.nms_iou_threshold[task_id],
                    pre_maxsize=test_cfg.nms.nms_pre_max_size[task_id],
                    post_max_size=test_cfg.nms.nms_post_max_size[task_id])
            else:
                raise NotImplementedError

            prediction_dict = {
                'box3d_lidar': selected_boxes,
                'scores': selected_scores,
                'label_preds': selected_labels
            }

            prediction_dicts.append(prediction_dict)

        return prediction_dicts 
    

class DetectionNetDense(DetectionNet):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def forward(self, example, return_loss=True):
        
        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )

        sp_tensor = self.reader(data)
        dense_tensor = sp_tensor.dense()
        conv_4, conv_5 = self.backbone(dense_tensor)
        
        # Neck → összefűzi és feldolgozza a feature mapeket
        neck_out = self.neck(conv_4, conv_5)

        output = self.bbox_head(neck_out)

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # with open(f'/PillarNet/PillarNet/quantized_outputs/{timestamp}.pkl', 'wb') as f:

        #     pickle.dump(output, f)

        if return_loss:

            return self.loss(example, output, self.test_cfg)
        
        else:

            return self.predict(example, output, self.test_cfg)

class DetectionNet_mod(DetectionNet):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def forward(self, example, return_loss=True):
        
        batch_size = len(example['metadata'])

        data = dict(
            points=example["points"],
            batch_size=batch_size,
        )

        sp_tensor = self.reader(data)
        dense_tensor = sp_tensor.dense()
        conv_4 = self.backbone(dense_tensor)
        
        # Neck → összefűzi és feldolgozza a feature mapeket
        neck_out = self.neck(conv_4)

        output = self.bbox_head(neck_out)


        if return_loss:

            return self.loss(example, output, self.test_cfg)
        
        else:

            return self.predict(example, output, self.test_cfg)