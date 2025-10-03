import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
import pickle
import math
import random
import os
import torch
import center_head

# Pillarizáció

class PillarFeatureNet(tf.keras.Model):
    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 64,
                 pillar_size: float = 0.075,
                 point_cloud_range: list = [-54, -54, -5.0, 54, 54, 3.0],
                 out_size=(180, 180),
                 dtype=tf.float32):
        """
        Args:
            pillar_size: float, pl. 0.32 (pillér mérete XY-ben)
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            in_channels: pont feature dimenzió (pl. intensity = 1)
            out_channels: BEV feature map csatornaszáma (pl. 64)
            out_size: (H, W) végső méret a hálóhoz
        """
        super().__init__(dtype=dtype)

        self.pillar_size = pillar_size
        self.pc_range = point_cloud_range
        self.out_size = out_size

        # rács paraméterek
        self.x_min, self.y_min, self.z_min, \
        self.x_max, self.y_max, self.z_max = point_cloud_range

        self.grid_x = int((self.x_max - self.x_min) / pillar_size)
        self.grid_y = int((self.y_max - self.y_min) / pillar_size)
        self.spatial_shape = (self.grid_y, self.grid_x)

        # MLP (Dense + ReLU)
        mlp_in = in_channels + 6
        self.fc = tf.keras.layers.Dense(out_channels, activation="relu")

    def call(self, xyz, point_features, training=False):
        """
        Args:
            xyz: (N, 3) pont koordináták (tf.Tensor)
            point_features: (N, C) pont feature (pl. intensity)

        Return:
            bev_map: (out_H, out_W, out_channels)
        """
        dtype = self.compute_dtype

        xyz = tf.cast(xyz, dtype)
        point_features = tf.cast(point_features, dtype)

        # --- Normalizálás [-1,1]-be ---
        x_norm = (xyz[:, 0] - self.x_min) / (self.x_max - self.x_min) * 2.0 - 1.0
        y_norm = (xyz[:, 1] - self.y_min) / (self.y_max - self.y_min) * 2.0 - 1.0
        z_norm = (xyz[:, 2] - self.z_min) / (self.z_max - self.z_min) * 2.0 - 1.0
        xyz = tf.stack([x_norm, y_norm, z_norm], axis=1)

        # Pillér indexek
        x_idx = tf.cast((x_norm + 1.0) / 2.0 * self.grid_x, tf.int32)
        y_idx = tf.cast((y_norm + 1.0) / 2.0 * self.grid_y, tf.int32)

        # Csak a tartományban lévő pontok
        mask = (x_idx >= 0) & (x_idx < self.grid_x) & \
               (y_idx >= 0) & (y_idx < self.grid_y)
        
        mask = tf.cast(mask, tf.bool)

        x_idx = tf.boolean_mask(x_idx, mask)
        y_idx = tf.boolean_mask(y_idx, mask)
        xyz = tf.boolean_mask(xyz, mask)
        point_features = tf.boolean_mask(point_features, mask)

        # delta_xyz (pillér középponthoz képest)
        x_center = (tf.cast(x_idx, dtype) + 0.5) / self.grid_x * 2.0 - 1.0
        y_center = (tf.cast(y_idx, dtype) + 0.5) / self.grid_y * 2.0 - 1.0

        delta_x = xyz[:, 0] - x_center
        delta_y = xyz[:, 1] - y_center
        delta_z = xyz[:, 2] - 0.0  # ground plane

        group_features = tf.concat(
            [point_features,
             xyz,
             tf.stack([delta_x, delta_y, delta_z], axis=1)],
            axis=1
        )

        # MLP
        pf = self.fc(group_features, training=training)

        # Pillér pooling (max)
        indices = y_idx * self.grid_x + x_idx
        bev_flat = tf.math.unsorted_segment_max(pf, indices, self.grid_x * self.grid_y)

        # ha marad -inf → 0
        bev_flat = tf.where(bev_flat <= -3.4e38,
                            tf.cast(0.0, dtype),
                            bev_flat)

        # reshape (grid_y, grid_x, C)
        bev = tf.reshape(bev_flat, (self.grid_y, self.grid_x, -1))

        # Resize a hálóhoz
        bev = tf.image.resize(bev, self.out_size, method="bilinear")

        return bev

# Modell architektúra
def relu6(x): return layers.ReLU(max_value=6.0)(x)

def conv6(x, f, k=3, s=1, conv_name = None, activation_name = None):
    x = layers.Conv2D(f, k, s, padding='same', use_bias=False,  name=conv_name)(x)
    x = layers.Activation(relu6, name=activation_name)(x)
    return x

def up6(x, f, up_name = None, conv_name=None, activation_name = None):
    x = layers.UpSampling2D(size=2, interpolation='bilinear', name=up_name)(x)
    x = layers.Conv2D(f, 3, 1, padding='same', use_bias=False,
                      name=conv_name)(x)
    x = layers.Activation(relu6, name=activation_name)(x)
    return x

def build_student_flatten_decoder_lil(input_shape, heads):
    """
    input_shape: (180,180,4)
    heads pl.: {'hm':10,'height':6,'dim':18,'rot':84,'vel':42,'iou':42}
    hm fej logitot ad (lineáris), a többiek lineáris regressziók.
    """
    inp = layers.Input(shape=input_shape, name='input_layer')

    # Encoder: 180 -> 6
    x = conv6(inp,  32, 5, 1)   # 180
    x = conv6(x,    32, 3, 2)   # 90
    x = conv6(x,    64, 3, 1)   # 90
    x = conv6(x,    64, 3, 2)   # 45
    x = conv6(x,    96, 3, 2)   # 23
    x = conv6(x, 64, 3, 2) 
    x = conv6(x, 64, 3, 2)


    # Kötelező Flatten + Dense (ReLU6)
    f = layers.Flatten()(x)                         # 6*6*128
    f = layers.Dense(6*6*64, use_bias=False)(f)
    f = relu6(f)
    x = layers.Reshape((6, 6, 64))(f)

    # Decoder: 6 -> 192 -> crop 180
    x = up6(x, 64)  # 12
    x = up6(x,  96)  # 24
    x = up6(x,  64)  # 48
    x = up6(x,  48)  # 96
    x = up6(x,  32)  # 192
    x = layers.Cropping2D(cropping=((6,6),(6,6)))(x)  # 180

    outputs = {}
    for name, ch in heads.items():
        outputs[name] = layers.Conv2D(ch, 1, 1, padding='same', use_bias=True, name=name)(x)
    return Model(inp, outputs, name='student_bev_flatten_relu6')


def build_student_flatten_decoder_bigger(input_shape, heads):
    inp = layers.Input(shape=input_shape, name='input_layer')

    # Encoder: 180 -> 6
    x = conv6(inp,   64, 5, 2, conv_name='conv2d', activation_name='re_lu')   # 180. Itt eredetileg 1 volt a stride
    x = conv6(x,    128, 3, 2, conv_name='conv2d_1', activation_name='re_lu_1')   # 90
    x = conv6(x,    128, 3, 2, conv_name='conv2d_2', activation_name='re_lu_2')   # 90. Itt eredetileg 1 volt a stride
    x = conv6(x,    192, 3, 2, conv_name='conv2d_3', activation_name='re_lu_3')   # 45
    x = conv6(x,    192, 3, 2, conv_name='conv2d_4', activation_name='re_lu_4')   # 23
    x = conv6(x,    256, 3, 2, conv_name='conv2d_5', activation_name='re_lu_5')   # 12
    x = conv6(x,    256, 3, 2, conv_name='conv2d_6', activation_name='re_lu_6')   # 6
    x = conv6(x, 256, 3, 2, conv_name='conv2d_7', activation_name='re_lu_7')

    # Flatten + Dense (ReLU6)
    f = layers.Flatten(name='flatten')(x)                         
    f = layers.Dense(250, activation=relu6, use_bias=False, name='dense')(f)
    f = layers.Dense(6*6*256, use_bias=False, name='dense_1')(f)
    x = layers.Reshape((6,6,256), name='reshape')(f)

    # Decoder: 6 -> 192 -> crop 180
    x = up6(x, 256, up_name='up_samplind2d', conv_name='conv2d_8', activation_name='re_lu_8')   # 12
    x = up6(x, 192, up_name='up_sampling2d_1', conv_name='conv2d_9', activation_name='re_lu_9')   # 24
    x = up6(x, 128, up_name='up_sampling2d_2', conv_name='conv2d_10', activation_name='re_lu_10')   # 48
    x = up6(x, 128, up_name='up_sampling2d_3', conv_name='conv2d_11', activation_name='re_lu_11')   # 96
    x = up6(x, 64, up_name='up_sampling2d_4', conv_name='conv2d_12', activation_name='re_lu_12')    # 192
    x = layers.Cropping2D(cropping=((6,6),(6,6)), name='cropping2d')(x)  # 180

    # outputs = {}
    # for name, ch in heads.items():
    #     outputs[name] = layers.Conv2D(
    #         ch, 1, 1, padding='same', use_bias=True, name=name
    #     )(x)

    outputs = {
        'hm': layers.Conv2D(heads['hm'], 1, 1, padding='same', use_bias=True, name='conv2d_15')(x),
        'reg': layers.Conv2D(heads['reg'], 1, 1, padding='same', use_bias=True, name='conv2d_17')(x),
        'height': layers.Conv2D(heads['height'], 1, 1, padding='same', use_bias=True, name='conv2d_14')(x),
        'dim': layers.Conv2D(heads['dim'], 1, 1, padding='same', use_bias=True, name='conv2d_13')(x),
        'rot': layers.Conv2D(heads['rot'], 1, 1, padding='same', use_bias=True, name='conv2d_18')(x),
        'vel': layers.Conv2D(heads['vel'], 1, 1, padding='same', use_bias=True, name='conv2d_19')(x),
        'iou': layers.Conv2D(heads['iou'], 1, 1, padding='same', use_bias=True, name='conv2d_16')(x),
    }

    return Model(inp, outputs, name='student_bev_flatten_relu6_big')

# Ha a teacher heatmap LOGIT, legyen True. Ha már sigmoid utáni, legyen False.
TEACHER_HM_IS_LOGIT = True

def torch_or_np_to_hwC(x):  # [C,H,W] -> (H,W,C) float32
    arr = x
    if hasattr(arr, 'detach'):  # torch.Tensor
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1]:  # [C,H,W]
        arr = np.transpose(arr, (1,2,0))
    return arr.astype('float32')

def build_teacher_dict(sample):
    t = {}
    for k in ['hm', 'reg', 'height','dim','rot','vel','iou']:
        arr = torch_or_np_to_hwC(sample[f'{k}_raw'])
        t[k] = arr  # (180,180,Ck)
    return t

def make_fg_mask_from_hm(hm_hwC, thr=0.10):
    x = tf.convert_to_tensor(hm_hwC, tf.float32)
    prob = tf.math.sigmoid(x) if TEACHER_HM_IS_LOGIT else x
    fg = tf.cast(
        tf.greater(tf.reduce_max(prob, axis=-1, keepdims=True), thr),
        tf.float32
    )
    return fg

code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0]
common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel': (2, 2), 'iou': (1, 2)}
pillar_size=0.075
pc_range=[-54, -54, -5.0, 54, 54, 3.0]

tasks = [
    dict(stride=8, class_names=["car"]),
    dict(stride=8, class_names=["truck", "construction_vehicle"]),
    dict(stride=8, class_names=["bus", "trailer"]),
    dict(stride=8, class_names=["barrier"]),
    dict(stride=8, class_names=["motorcycle", "bicycle"]),
    dict(stride=8, class_names=["pedestrian", "traffic_cone"]),
    ]

class config:

        def __init__(self, post_center_limit_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                        max_per_img = 500,
                        nms = dict(
                        use_rotate_nms=True,
                        use_multi_class_nms=False,
                        nms_pre_max_size=1000,
                        nms_post_max_size=83,
                        nms_iou_threshold=0.2,
                    ),
                    score_threshold = 0.1,
                    rectifier = 0.5,
                    pc_range = pc_range[:2],
                    out_size_factor = 8,
                    voxel_size=[pillar_size, pillar_size],
                    double_flip = False,
                    return_raw = False,
                    per_class_nms = False,
                    circular_nms = False):
            
            self.post_center_limit_range = post_center_limit_range
            self.max_per_img = max_per_img
            self.nms = nms
            self.score_threshold = score_threshold
            self.rectifier = rectifier
            self.pc_range = pc_range
            self.out_size_factor = out_size_factor
            self.voxel_size = voxel_size
            self.double_flip = double_flip
            self.return_raw = return_raw
            self.per_class_nms = per_class_nms
            self.circular_nms = circular_nms

test_cfg = config()

class MapDistillTrainer(tf.keras.Model):
    def __init__(self, student, w, use_tp_fp_fn, lambda_1, lambda_2, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.w = w
        self.mae = tf.keras.losses.MeanAbsoluteError()
        # HM: ha a teacher HM logit, from_logits=True és a target is logit kompatibilis (BCE logit-logit).
        # Egyszerű és stabil alternatíva: alakítsd a teacher HM-et probbá, és from_logits=True marad a studentnél.
        self.bce_from_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.use_tp_fp_fn = use_tp_fp_fn
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def kl_divergence(self, student_logits, teacher_logits, T=2.0):
        """
        KL divergence distillation loss.
        student_logits: (B,H,W,C)
        teacher_logits: (B,H,W,C)
        """
        teacher_probs = tf.nn.softmax(teacher_logits / T, axis=-1)
        student_log_probs = tf.nn.log_softmax(student_logits / T, axis=-1)
        return tf.reduce_mean(
        tf.reduce_sum(teacher_probs * (tf.math.log(teacher_probs + 1e-8) - student_log_probs), axis=-1)
        ) * (T * T)
    
    def make_tp_fp_fn_masks(self, hm_gt, hm_pred, thr=0.1):
        gt = tf.cast(hm_gt > thr, tf.float32)
        pred = tf.cast(tf.sigmoid(hm_pred) > thr, tf.float32)

        tp = gt * pred                  # mindkettő jelzi → TP
        fn = gt * (1 - pred)            # GT jelzi, pred nem → FN
        fp = (1 - gt) * pred            # GT nem, pred jelzi → FP

        return tp, fn, fp


    @tf.function
    def train_step(self, data):

        bev = data['bev']         
        teacher = data['teacher']                                # dict of (B,180,180,Ck)
        mask_fg = data.get('mask_fg', None)                      # (B,180,180,1)

        with tf.GradientTape() as tape:
            out = self.student(bev, training=True)
            loss = 0.0

            # Heatmap (teacher probbá alakítva a stabilitás kedvéért)
            hm_t = teacher['hm']
            hm_t_prob = tf.sigmoid(hm_t) if TEACHER_HM_IS_LOGIT else hm_t
            hm_s_logits = out['hm']
            hm_s_prob = tf.sigmoid(hm_s_logits)

            # Így nézett ki ereddetileg a loss számítás

            # if self.use_tp_fp_fn:
            #     tp, fn, fp = self.make_tp_fp_fn_masks(hm_t_prob, hm_s_logits, thr=0.1)
            #     w = tf.stop_gradient(1.0 * (tp + fn) + 0.25 * fp)

            # elif mask_fg is not None:
            #     w = tf.stop_gradient(mask_fg)

            # else:
            #     w = None

            # if w is not None:
            #     # maszkolt BCE
            #     bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=hm_t_prob, logits=hm_s_logits)
            #     l_hm = tf.reduce_sum(bce * w) / (tf.reduce_sum(w) + 1e-6)

            #     # maszkolt KL
            #     teacher_probs = tf.nn.softmax(hm_t / 2.0, axis=-1)
            #     student_log_probs = tf.nn.log_softmax(hm_s_logits / 2.0, axis=-1)
            #     kl = tf.reduce_sum(
            #         teacher_probs * (tf.math.log(teacher_probs + 1e-8) - student_log_probs),
            #         axis=-1, keepdims=True
            #     )
            #     l_hm_kl = tf.reduce_sum(kl * w) / (tf.reduce_sum(w) + 1e-6) * (2.0 * 2.0)
            # else:
            #     # fallback: teljes térképen
            #     l_hm = self.bce_from_logits(hm_t_prob, hm_s_logits)
            #     l_hm_kl = self.kl_divergence(hm_s_logits, hm_t, T=2.0)

            # loss += 0.5 * self.w.get('hm', 1.0) * (l_hm + l_hm_kl)

            # # Maszkos L1 a regressziókra
            # def masked_l1(t, s):
            #     if mask_fg is None:
            #         return self.mae(t, s)
            #     w = tf.stop_gradient(mask_fg)
            #     num = tf.reduce_sum(tf.abs(t - s) * w)
            #     den = tf.reduce_sum(w) + 1e-6
            #     return num / den

            # for name in ['reg', 'height','dim','rot','vel','iou']:
            #     if name in out and name in teacher and name in self.w:
            #         loss += 0.5 * self.w[name] * masked_l1(teacher[name], out[name])

            # Új loss számítás:

            student_mask = make_fg_mask_from_hm(hm_s_logits)

            mask_bce_loss = tf.keras.losses.BinaryCrossentropy(
                from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            )(mask_fg, student_mask)

            loss += self.lambda_1*mask_bce_loss * 1.0
            
            tp_regions = mask_fg * student_mask

            if tf.reduce_sum(tp_regions) > 0:

                teacher_max_channel_idx_per_pixel = tf.argmax(hm_t_prob, axis=-1)
                
                sample_weights = tf.stop_gradient(tp_regions)

                channel_classification_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits = False,
                    reduction = tf.keras.losses.Reduction.NONE
                )(teacher_max_channel_idx_per_pixel, hm_s_prob)
                weighted_channel_loss = tf.reduce_sum(tf.expand_dims(channel_classification_loss, axis=-1) * sample_weights) / tf.maximum(1.0, tf.reduce_sum(sample_weights))

                loss += self.lambda_1*weighted_channel_loss * 1.0
                
            for name in ['reg', 'height','dim','rot','vel','iou']:

                if name in out and name in teacher and name in self.w:
                                        
                    #mse_loss_per_pixel = tf.keras.losses.MeanSquaredError(
                    #    reduction=tf.keras.losses.Reduction.NONE
                    #)(teacher[name], out[name])

                    mse_loss_per_pixel = tf.square(teacher[name] - out[name])

                    C = teacher[name].shape[-1]
                    mask_new = tf.tile(mask_fg, [1,1,1,C])
                    weighted_mse_loss_per_pixel = mse_loss_per_pixel * mask_new

                    head_mse_loss = tf.reduce_sum(weighted_mse_loss_per_pixel) / tf.reduce_sum(mask_fg)

                    loss += self.lambda_1*self.w[name] * head_mse_loss
                    
            # preds_dicts = make_preds_dicts_from_student(student_outputs=out, tasks=tasks)
            # example = make_example_from_teacher(teacher_outputs=teacher, tasks=tasks, mask_fg=mask_fg)

            # loss_label = self.cent_head.loss(example=example, preds_dicts=preds_dicts, test_cfg = test_cfg)

            # for k, v in loss_label.items():
            #     # v lista, általában 1 elem vagy batch-nyi tensor
            #     # összegzés / átlagolás kell
            #     val = tf.reduce_mean(tf.concat([tf.reshape(x, [-1]) for x in v], axis=0))
            #     loss += self.lambda_2 * val

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.train_loss.update_state(loss)
        
        return {"loss": self.train_loss.result()}
    
# Dataset

def make_streaming_dataset_pillars(tokens, point_folder, pickle_folder, heads, batch_size=16, shuffle=True):
    ch = {k: heads[k] for k in ['hm','reg','height','dim','rot','vel','iou']}

    def gen():
        idxs = np.arange(len(tokens))
        if shuffle:
            np.random.shuffle(idxs)

        for s in range(0, len(tokens), batch_size):
            sel = idxs[s:s+batch_size]

            bevs, masks, tokenss = [], [], []
            teach = {k: [] for k in ch.keys()}
        
            for j in sel:
                smp = tokens[j]
                # pts = np.fromfile(os.path.join(lidar_folder, lidar_paths[smp]), dtype=np.float32).reshape(-1, 5)  # (N,5): [x,y,z,intensity,time]

                # bev = pillar_net(pts[:, :3], pts[:, 3:5], training=False).numpy()

                with open(os.path.join(point_folder, smp+'.pkl'), 'rb') as f:

                    bev = pickle.load(f)

                bevs.append(torch_or_np_to_hwC(bev[smp]))

                with open(os.path.join(pickle_folder, smp+'.pkl'), 'rb') as f:

                    teacher = pickle.load(f)

                t = build_teacher_dict(teacher)
                m = tf.cast(make_fg_mask_from_hm(t["hm"]), tf.float32)
                masks.append(m)
                tokenss.append(smp)

                for k in ch.keys():
                    teach[k].append(t[k].astype("float32"))

            # konvertálás RaggedTensor-ba közvetlenül
            batch = {
                "bev": np.stack(bevs, axis=0).astype(np.float32),
                "teacher": {k: np.stack(teach[k], axis=0) for k in ch.keys()},
                "mask_fg": np.stack(masks, axis=0),
                "tokenss": np.array(tokenss)
            }
            yield batch

    output_signature = {
        "bev": tf.TensorSpec(shape=(None, 1440, 1440, 32), dtype=tf.float32),
        "teacher": {
            "hm": tf.TensorSpec(shape=(None, 180, 180, ch["hm"]), dtype=tf.float32),
            "reg": tf.TensorSpec(shape=(None, 180, 180, ch["reg"]), dtype=tf.float32),
            "height": tf.TensorSpec(shape=(None, 180, 180, ch["height"]), dtype=tf.float32),
            "dim": tf.TensorSpec(shape=(None, 180, 180, ch["dim"]), dtype=tf.float32),
            "rot": tf.TensorSpec(shape=(None, 180, 180, ch["rot"]), dtype=tf.float32),
            "vel": tf.TensorSpec(shape=(None, 180, 180, ch["vel"]), dtype=tf.float32),
            "iou": tf.TensorSpec(shape=(None, 180, 180, ch["iou"]), dtype=tf.float32),
        },
        "mask_fg": tf.TensorSpec(shape=(None, 180, 180, 1), dtype=tf.float32),
        "tokenss": tf.TensorSpec(shape=(None, ), dtype=tf.string)
    }
    
    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature
    ).repeat().prefetch(tf.data.AUTOTUNE)

# def make_example_from_teacher(teacher_outputs, tasks, mask_fg=None):
#     """
#     Teacher kimenetből (összes head egyben) csinál example szótárat.

#     Args:
#         teacher_outputs: dict, pl. {"hm": tf.Tensor[B,H,W,C], "reg": ..., ...}
#         tasks: list of dict, pl. a te tasks változód
#         mask_fg: opcionális bináris maszk (B,H,W,1).
#                  Ha nincs, minden pixel aktív lesz minden taskra.

#     Returns:
#         example: dict, kulcs → list[task_id] → tensor
#     """
#     example = {}

#     # minden head feldarabolása taskokra
#     for head_name, full_tensor in teacher_outputs.items():
#         B, H, W, C = full_tensor.shape
#         task_tensors = []
#         start_c = 0

#         for t_id, t in enumerate(tasks):
#             if head_name == "hm":
#                 num_c = len(t["class_names"])
#             elif head_name in ["reg", "rot", "vel"]:
#                 num_c = 2
#             elif head_name == "height":
#                 num_c = 1
#             elif head_name == "dim":
#                 num_c = 3
#             elif head_name == "iou":
#                 num_c = len(t["class_names"])
#             else:
#                 raise ValueError(f"Ismeretlen head: {head_name}")

#             task_tensor = full_tensor[..., start_c:start_c+num_c]
#             task_tensors.append(task_tensor)
#             start_c += num_c

#         example[head_name] = task_tensors

#     # maszkok taskonként
#     if mask_fg is not None:
#         example["mask"] = [mask_fg for _ in tasks]
#     else:
#         example["mask"] = [tf.ones_like(x[..., :1], dtype=tf.float32)
#                            for x in example["hm"]]

#     # ind és cat a maszk alapján
#     example["ind"] = []
#     example["cat"] = []

#     for t_id, (hm_task, mask_task) in enumerate(zip(example["hm"], example["mask"])):
#         B = tf.shape(hm_task)[0]
#         H = tf.shape(hm_task)[1]
#         W = tf.shape(hm_task)[2]
#         mask_flat = tf.reshape(mask_task, [B, -1])  # (B, H*W)

#         inds_per_batch = []
#         cats_per_batch = []
#         for b in range(B):
#             flat_inds = tf.boolean_mask(tf.range(H*W), mask_flat[b] > 0)
#             inds_per_batch.append(flat_inds)

#             flat_hm = tf.reshape(hm_task[b], [H*W, -1])  # (H*W, C_task)
#             cat_ids = tf.argmax(tf.boolean_mask(flat_hm, mask_flat[b] > 0), axis=-1)
#             cats_per_batch.append(cat_ids)

#         # pad batch-szinten
#         max_len = tf.reduce_max([tf.shape(x)[0] for x in inds_per_batch])
#         inds_pad = [tf.pad(x, [[0, max_len - tf.shape(x)[0]]]) for x in inds_per_batch]
#         cats_pad = [tf.pad(x, [[0, max_len - tf.shape(x)[0]]]) for x in cats_per_batch]

#         inds = tf.stack(inds_pad, axis=0)
#         cats = tf.stack(cats_pad, axis=0)

#         example["ind"].append(inds)
#         example["cat"].append(cats)

#     # anno_box és gt_box csak egyszer kell
#     example["anno_box"] = []
#     example["gt_box"] = []
    
#     if "vel" in teacher_outputs:
#         anno_box = tf.concat([
#             teacher_outputs["reg"],     # (B,H,W,2)
#             teacher_outputs["height"],  # (B,H,W,1)
#             teacher_outputs["dim"],     # (B,H,W,3)
#             teacher_outputs["vel"],     # (B,H,W,2)
#             teacher_outputs["rot"]      # (B,H,W,2)
#         ], axis=-1)
#     else:
#         anno_box = tf.concat([
#             teacher_outputs["reg"],
#             teacher_outputs["height"],
#             teacher_outputs["dim"],
#             teacher_outputs["rot"]
#         ], axis=-1)

#     start_c = 0
#     for t in tasks:
#         num_c = 0
#         num_c += 2  # reg
#         num_c += 1  # height
#         num_c += 3  # dim
#         if "vel" in teacher_outputs:
#             num_c += 2
#         num_c += 2  # rot

#         task_anno = anno_box[..., start_c:start_c+num_c]
#         example["anno_box"].append(task_anno)
#         start_c += num_c

    
#     for t in range(len(tasks)):

#         batch_dim = tf.exp(tf.clip_by_value(example['dim'][t], -5, 5))
#         #batch_dim = tf.transpose(batch_dim, [0, 2, 3, 1])
#         batch_rot = example['rot'][t]  # tf.transpose(teacher_outputs['rot'], [0, 2, 3, 1])
#         batch_rots = batch_rot[..., 0:1]
#         batch_rotc = batch_rot[..., 1:2]
#         # batch_reg = tf.transpose(teacher_outputs['reg'], [0, 2, 3, 1])
#         # batch_hei = tf.transpose(teacher_outputs['height'], [0, 2, 3, 1])
#         batch_rot = tf.atan2(batch_rots, batch_rotc)

#         batch, H, W, _ = batch_dim.shape

#         batch_reg = tf.reshape(example['reg'][t], [batch, H * W, 2])
#         batch_hei = tf.reshape(example['height'][t], [batch, H * W, 1])
#         batch_rot = tf.reshape(batch_rot, [batch, H * W, 1])
#         batch_dim = tf.reshape(batch_dim, [batch, H * W, 3])

#         ys, xs = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
#         ys = tf.cast(ys[None, ...], batch_dim.dtype)
#         xs = tf.cast(xs[None, ...], batch_dim.dtype)

#         xs = tf.reshape(xs, [1, -1, 1]) + batch_reg[:, :, 0:1]
#         ys = tf.reshape(ys, [1, -1, 1]) + batch_reg[:, :, 1:2]

#         xs = xs * int(8) * 0.075 + (-54)
#         ys = ys * int(8) * 0.075 + (-54)
#         batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_rot], axis=2)
#         #batch_box_preds = tf.transpose(batch_box_preds, [0, 2, 1])
#         batch_box_preds = tf.reshape(batch_box_preds, [batch, H, W, -1])

#         example['gt_box'].append(batch_box_preds)

#     return example

def make_example_from_teacher(teacher_outputs, tasks, mask_fg=None):
    """
    Teacher kimenetből (összes head egyben) csinál example szótárat.

    Args:
        teacher_outputs: dict, pl. {"hm": tf.Tensor[B,H,W,C], "reg": ..., ...}
        tasks: list of dict, pl. a te tasks változód
        mask_fg: opcionális bináris maszk (B,H,W,1).
                 Ha nincs, minden pixel aktív lesz minden taskra.

    Returns:
        example: dict, kulcs → list[task_id] → tensor
    """
    example = {}

    # minden head feldarabolása taskokra
    for head_name, full_tensor in teacher_outputs.items():
        B = tf.shape(full_tensor)[0]
        H = tf.shape(full_tensor)[1]
        W = tf.shape(full_tensor)[2]
        C = tf.shape(full_tensor)[3]

        task_tensors = []
        start_c = 0

        for t in tasks:
            if head_name == "hm":
                num_c = len(t["class_names"])
            elif head_name in ["reg", "rot", "vel"]:
                num_c = 2
            elif head_name == "height":
                num_c = 1
            elif head_name == "dim":
                num_c = 3
            elif head_name == "iou":
                num_c = len(t["class_names"])
            else:
                raise ValueError(f"Ismeretlen head: {head_name}")

            task_tensor = full_tensor[..., start_c:start_c+num_c]
            task_tensors.append(task_tensor)
            start_c += num_c

        example[head_name] = task_tensors

    # maszkok taskonként
    if mask_fg is not None:
        example["mask"] = [mask_fg for _ in tasks]
    else:
        example["mask"] = [tf.ones_like(x[..., :1], dtype=tf.float32)
                           for x in example["hm"]]

    # ind és cat a maszk alapján
    example["ind"] = []
    example["cat"] = []

    for hm_task, mask_task in zip(example["hm"], example["mask"]):
        B = int(tf.shape(hm_task)[0])
        H = int(tf.shape(hm_task)[1])
        W = int(tf.shape(hm_task)[2])
        C_task = int(tf.shape(hm_task)[-1])

        mask_flat = tf.reshape(mask_task, [B, -1])  # (B, H*W)

        def process_one(mask_b, hm_b):
            # mask_b: (H*W,)
            # hm_b: (H,W,C_task)
            flat_hm = tf.reshape(hm_b, [-1, C_task])  # (H*W, C_task)
            flat_inds = tf.boolean_mask(tf.range(H*W), mask_b > 0)
            cat_ids = tf.cast(tf.argmax(tf.boolean_mask(flat_hm, mask_b > 0), axis=-1), tf.int32)
            return flat_inds, cat_ids

        inds, cats = tf.map_fn(
            lambda args: process_one(args[0], args[1]),
            (mask_flat, hm_task),
            fn_output_signature=(
                tf.RaggedTensorSpec(shape=[None], dtype=tf.int32),
                tf.RaggedTensorSpec(shape=[None], dtype=tf.int32)
            )
        )

        inds_ragged = tf.ragged.stack(inds)  # (B, None)
        cats_ragged = tf.ragged.stack(cats)  # (B, None)

        # pad ragged → dense (mint az eredetiben)
        max_len = int(tf.reduce_max(inds_ragged.row_lengths()))
        inds = inds_ragged.to_tensor(default_value=0)
        cats = cats_ragged.to_tensor(default_value=0)

        inds = tf.reshape(inds, [tf.shape(inds)[0], -1])
        cats = tf.reshape(cats, [tf.shape(cats)[0], -1])

        example["ind"].append(inds)
        example["cat"].append(cats)

    # anno_box és gt_box
    example["anno_box"] = []
    example["gt_box"] = []

    if "vel" in teacher_outputs:
        anno_box = tf.concat([
            teacher_outputs["reg"],     # (B,H,W,2)
            teacher_outputs["height"],  # (B,H,W,1)
            teacher_outputs["dim"],     # (B,H,W,3)
            teacher_outputs["vel"],     # (B,H,W,2)
            teacher_outputs["rot"]      # (B,H,W,2)
        ], axis=-1)
    else:
        anno_box = tf.concat([
            teacher_outputs["reg"],
            teacher_outputs["height"],
            teacher_outputs["dim"],
            teacher_outputs["rot"]
        ], axis=-1)

    start_c = 0
    for t in tasks:
        num_c = 2 + 1 + 3  # reg + height + dim
        if "vel" in teacher_outputs:
            num_c += 2
        num_c += 2  # rot

        task_anno = anno_box[..., start_c:start_c+num_c]
        example["anno_box"].append(task_anno)
        start_c += num_c

    # gt_box: ugyanaz, mint a loss-ban építve
    for t in range(len(tasks)):
        batch_dim = tf.exp(tf.clip_by_value(example['dim'][t], -5, 5))
        batch_rot = example['rot'][t]
        batch_rots = batch_rot[..., 0:1]
        batch_rotc = batch_rot[..., 1:2]
        batch_rot = tf.atan2(batch_rots, batch_rotc)

        B = tf.shape(batch_dim)[0]
        H = tf.shape(batch_dim)[1]
        W = tf.shape(batch_dim)[2]

        batch_reg = tf.reshape(example['reg'][t], [B, H * W, 2])
        batch_hei = tf.reshape(example['height'][t], [B, H * W, 1])
        batch_rot = tf.reshape(batch_rot, [B, H * W, 1])
        batch_dim = tf.reshape(batch_dim, [B, H * W, 3])

        ys, xs = tf.meshgrid(tf.range(H), tf.range(W), indexing="ij")
        ys = tf.cast(ys[None, ...], batch_dim.dtype)
        xs = tf.cast(xs[None, ...], batch_dim.dtype)

        xs = tf.reshape(xs, [1, -1, 1]) + batch_reg[:, :, 0:1]
        ys = tf.reshape(ys, [1, -1, 1]) + batch_reg[:, :, 1:2]

        xs = xs * int(8) * 0.075 + (-54)
        ys = ys * int(8) * 0.075 + (-54)
        batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_rot], axis=2)
        batch_box_preds = tf.reshape(batch_box_preds, [B, H, W, -1])

        example['gt_box'].append(batch_box_preds)

    return example


def make_preds_dicts_from_student(student_outputs, tasks):
    """
    Student kimenetből (összes head egyben) csinál preds_dicts listát.
    
    Args:
        student_outputs: dict, pl. {"hm": tf.Tensor[B,H,W,C], "reg": ..., ...}
        tasks: ugyanaz mint fent
    
    Returns:
        preds_dicts: list[task_id] → dict head_name → tensor
    """
    preds_dicts = []
    B, H, W, _ = student_outputs["hm"].shape

    for t_id, t in enumerate(tasks):
        preds_dict = {}
        start_c = 0

        for head_name, full_tensor in student_outputs.items():
            if head_name == "hm":
                num_c = len(t["class_names"])
            elif head_name in ["reg", "rot", "vel"]:
                num_c = 2
            elif head_name == "height":
                num_c = 1
            elif head_name == "dim":
                num_c = 3
            elif head_name == "iou":
                num_c = len(t["class_names"])
            else:
                raise ValueError(f"Ismeretlen head: {head_name}")

            preds_dict[head_name] = full_tensor[..., start_c:start_c+num_c]
            start_c += num_c

        preds_dicts.append(preds_dict)

    return preds_dicts