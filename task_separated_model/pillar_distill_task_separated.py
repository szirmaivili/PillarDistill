import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras import Sequential
import pickle
import math
import random
import os
import torch
import center_head
import centernet_loss
tf.config.run_functions_eagerly(True)

# Modell architektúra
# def relu6(x): return layers.ReLU(max_value=6.0)(x)

# def conv6(x, f, k=3, s=1, conv_name = None, activation_name = None):
#     x = layers.Conv2D(f, k, s, padding='same', use_bias=False,  name=conv_name)(x)
#     x = layers.Activation(relu6, name=activation_name)(x)
#     return x

# def up6(x, f, up_name = None, conv_name=None, activation_name = None):
#     x = layers.UpSampling2D(size=2, interpolation='bilinear', name=up_name)(x)
#     x = layers.Conv2D(f, 3, 1, padding='same', use_bias=False,
#                       name=conv_name)(x)
#     x = layers.Activation(relu6, name=activation_name)(x)
#     return x

def relu6(x):
    return layers.ReLU(max_value=6.0)(x)


def conv6_layers(f, k=3, s=1, conv_name=None, activation_name=None):
    return [
        layers.Conv2D(f, k, strides=s, padding='same', use_bias=False, name=conv_name),
        layers.Activation(relu6, name=activation_name),
    ]


def up6_layers(f, up_name=None, conv_name=None, activation_name=None):
    return [
        layers.UpSampling2D(size=2, interpolation='bilinear', name=up_name),
        layers.Conv2D(f, 3, 1, padding='same', use_bias=False, name=conv_name),
        layers.Activation(relu6, name=activation_name),
    ]



# def build_student_flatten_decoder_bigger(input_shape, heads):
#     inp = layers.Input(shape=input_shape, name='input_layer')

#     # Encoder: 180 -> 6
#     x = conv6(inp,   64, 5, 2, conv_name='conv2d', activation_name='re_lu')   # 180. Itt eredetileg 1 volt a stride
#     x = conv6(x,    128, 3, 2, conv_name='conv2d_1', activation_name='re_lu_1')   # 90
#     x = conv6(x,    128, 3, 2, conv_name='conv2d_2', activation_name='re_lu_2')   # 90. Itt eredetileg 1 volt a stride
#     x = conv6(x,    192, 3, 2, conv_name='conv2d_3', activation_name='re_lu_3')   # 45
#     x = conv6(x,    192, 3, 2, conv_name='conv2d_4', activation_name='re_lu_4')   # 23
#     x = conv6(x,    256, 3, 2, conv_name='conv2d_5', activation_name='re_lu_5')   # 12
#     x = conv6(x,    256, 3, 2, conv_name='conv2d_6', activation_name='re_lu_6')   # 6
#     x = conv6(x, 256, 3, 2, conv_name='conv2d_7', activation_name='re_lu_7')

#     # Flatten + Dense (ReLU6)
#     f = layers.Flatten(name='flatten')(x)                         
#     f = layers.Dense(250, activation=relu6, use_bias=False, name='dense')(f)
#     f = layers.Dense(6*6*256, use_bias=False, name='dense_1')(f)
#     x = layers.Reshape((6,6,256), name='reshape')(f)

#     # Decoder: 6 -> 192 -> crop 180
#     x = up6(x, 256, up_name='up_samplind2d', conv_name='conv2d_8', activation_name='re_lu_8')   # 12
#     x = up6(x, 192, up_name='up_sampling2d_1', conv_name='conv2d_9', activation_name='re_lu_9')   # 24
#     x = up6(x, 128, up_name='up_sampling2d_2', conv_name='conv2d_10', activation_name='re_lu_10')   # 48
#     x = up6(x, 128, up_name='up_sampling2d_3', conv_name='conv2d_11', activation_name='re_lu_11')   # 96
#     x = up6(x, 64, up_name='up_sampling2d_4', conv_name='conv2d_12', activation_name='re_lu_12')    # 192
#     x = layers.Cropping2D(cropping=((6,6),(6,6)), name='cropping2d')(x)  # 180

#     # outputs = {}
#     # for name, ch in heads.items():
#     #     outputs[name] = layers.Conv2D(
#     #         ch, 1, 1, padding='same', use_bias=True, name=name
#     #     )(x)

#     outputs = {
#         'hm': layers.Conv2D(heads['hm'], 1, 1, padding='same', use_bias=True, name='conv2d_15')(x),
#         'reg': layers.Conv2D(heads['reg'], 1, 1, padding='same', use_bias=True, name='conv2d_17')(x),
#         'height': layers.Conv2D(heads['height'], 1, 1, padding='same', use_bias=True, name='conv2d_14')(x),
#         'dim': layers.Conv2D(heads['dim'], 1, 1, padding='same', use_bias=True, name='conv2d_13')(x),
#         'rot': layers.Conv2D(heads['rot'], 1, 1, padding='same', use_bias=True, name='conv2d_18')(x),
#         'vel': layers.Conv2D(heads['vel'], 1, 1, padding='same', use_bias=True, name='conv2d_19')(x),
#         'iou': layers.Conv2D(heads['iou'], 1, 1, padding='same', use_bias=True, name='conv2d_16')(x),
#     }

#     return Model(inp, outputs, name='student_bev_flatten_relu6_big')

def build_student_flatten_decoder_bigger(input_shape, dimensions):
    base_layers = [
        layers.InputLayer(input_shape=input_shape, name='input_layer'),
    ]

    base_layers.extend(conv6_layers(64, 5, 2, conv_name='conv2d', activation_name='re_lu'))
    base_layers.extend(conv6_layers(128, 3, 2, conv_name='conv2d_1', activation_name='re_lu_1'))
    base_layers.extend(conv6_layers(128, 3, 2, conv_name='conv2d_2', activation_name='re_lu_2'))
    base_layers.extend(conv6_layers(192, 3, 2, conv_name='conv2d_3', activation_name='re_lu_3'))
    base_layers.extend(conv6_layers(192, 3, 2, conv_name='conv2d_4', activation_name='re_lu_4'))
    base_layers.extend(conv6_layers(256, 3, 2, conv_name='conv2d_5', activation_name='re_lu_5'))
    base_layers.extend(conv6_layers(256, 3, 2, conv_name='conv2d_6', activation_name='re_lu_6'))
    base_layers.extend(conv6_layers(256, 3, 2, conv_name='conv2d_7', activation_name='re_lu_7'))

    base_layers.append(layers.Flatten(name='flatten'))
    base_layers.append(layers.Dense(250, activation=relu6, use_bias=False, name='dense'))
    base_layers.append(layers.Dense(6 * 6 * 256, use_bias=False, name='dense_1'))
    base_layers.append(layers.Reshape((6, 6, 256), name='reshape'))

    base_layers.extend(up6_layers(256, up_name='up_samplind2d', conv_name='conv2d_8', activation_name='re_lu_8'))
    base_layers.extend(up6_layers(192, up_name='up_sampling2d_1', conv_name='conv2d_9', activation_name='re_lu_9'))
    base_layers.extend(up6_layers(128, up_name='up_sampling2d_2', conv_name='conv2d_10', activation_name='re_lu_10'))
    base_layers.extend(up6_layers(128, up_name='up_sampling2d_3', conv_name='conv2d_11', activation_name='re_lu_11'))
    base_layers.extend(up6_layers(64, up_name='up_sampling2d_4', conv_name='conv2d_12', activation_name='re_lu_12'))

    base_model = Sequential(base_layers, name='student_bev_flatten_relu6_big_base')

    heads_input = layers.Input(shape=base_model.output_shape[1:], name='heads_input')
    x = layers.Cropping2D(cropping=((6, 6), (6, 6)), name='cropping2d')(heads_input)

    head_names = list(dimensions[0].keys())

    outputs = {}
    for head_name in head_names:
        task_outputs = []
        for task_idx, task_dims in enumerate(dimensions):
            head_dim = task_dims[head_name]
            head_out = layers.Conv2D(
                head_dim,
                1,
                1,
                padding='same',
                use_bias=True,
                name=f'{head_name}_task{task_idx}_head'
            )(x)
            task_outputs.append(head_out)

        # Fejen belül: a taskok csatornánként összefűzése
        merged_head = layers.Concatenate(axis=-1, name=f'{head_name}_merged')(task_outputs)
        outputs[head_name] = merged_head

    # outputs = {
    #     'hm': layers.Conv2D(heads['hm'], 1, 1, padding='same', use_bias=True, name='conv2d_15')(x),
    #     'reg': layers.Conv2D(heads['reg'], 1, 1, padding='same', use_bias=True, name='conv2d_17')(x),
    #     'height': layers.Conv2D(heads['height'], 1, 1, padding='same', use_bias=True, name='conv2d_14')(x),
    #     'dim': layers.Conv2D(heads['dim'], 1, 1, padding='same', use_bias=True, name='conv2d_13')(x),
    #     'rot': layers.Conv2D(heads['rot'], 1, 1, padding='same', use_bias=True, name='conv2d_18')(x),
    #     'vel': layers.Conv2D(heads['vel'], 1, 1, padding='same', use_bias=True, name='conv2d_19')(x),
    #     'iou': layers.Conv2D(heads['iou'], 1, 1, padding='same', use_bias=True, name='conv2d_16')(x),
    # }

    heads_model = Model(heads_input, outputs, name='student_bev_flatten_relu6_big_heads')
    full_model = tf.keras.Model(inputs=base_model.input,
                                outputs=heads_model(base_model.output),
                                name='student_bev_flatten_relu6_big') #Sequential([base_model, heads_model], name='student_bev_flatten_relu6_big')

    return full_model, base_model, heads_model

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

code_weights=tf.convert_to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0], dtype=tf.float32)
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
focal_loss = centernet_loss.FastFocalLoss()

def raw_preds_2_preds_dicts(raw_preds: dict, dimensions: dict):
    # Batch méretű objektumok: B, H, W, C. NEM KELL A DIMENZIÓKAT PERMUTÁLNI
    preds_dicts = []

    # hm hozzáadása:
    preds_dicts.append({'hm': raw_preds['hm'][..., 0:1]})
    preds_dicts.append({'hm': raw_preds['hm'][..., 1:3]})
    preds_dicts.append({'hm': raw_preds['hm'][..., 3:5]})
    preds_dicts.append({'hm': raw_preds['hm'][..., 5:6]})
    preds_dicts.append({'hm': raw_preds['hm'][..., 6:8]})
    preds_dicts.append({'hm': raw_preds['hm'][..., 8:10]})

    # Többi fej hozzáadása: 
    for i in range(6):

        for key in dimensions.keys():

            preds_dicts[i].update({key: raw_preds[key][..., i*dimensions[key]//6 : (i+1)*dimensions[key]//6]})

    return preds_dicts

dimensions = {'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}

@tf.autograph.experimental.do_not_convert
def set_example_shapes(ex):
    """
    Végigmegy az example dictionary minden kulcsán és elemén,
    és beállítja a fix alakokat a TensorFlow számára.
    """
    for key, tensor_list in ex.items():
        new_list = []
        for i, t in enumerate(tensor_list):
            if isinstance(t, np.ndarray):
                t = tf.convert_to_tensor(t, dtype=tf.float32)
            if key == "gt_boxes_and_cls":
                t.set_shape([None, 10])
            elif key == "hm":
                # hm eltérő csatornaszámú lehet, ezért None az utolsó dimenzió
                t.set_shape([None, 180, 180, None])
            elif key == "anno_box":
                t.set_shape([None, 500, 10])
            elif key in ["ind", "mask", "cat"]:
                t.set_shape([None, 500])
            elif key == "gt_box":
                t.set_shape([None, 500, 7])
            else:
                # fallback: nem ismert kulcs
                t.set_shape([None, None])

def load_example_py(tokens):
    examples = []
    for t in tokens.numpy():
        t_str = t.decode("utf-8")
        with open(os.path.join(r'/PillarDistill/teacher_examples', t_str + ".pkl"), "rb") as f:
            examples.append(pickle.load(f))

    num_tasks = 6
    keys = list(examples[0].keys())
    outs = []
    for key in keys:
        key_group = []
        for task_id in range(num_tasks):
            task_spec = [examples[b][key][task_id] for b in range(len(examples))]
            key_group.append(np.stack(task_spec, axis=0).astype(np.float32))
        outs.append(key_group)
    return outs

class MapDistillTrainer(tf.keras.Model):
    def __init__(self, student, w, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.w = w
        self.mae = tf.keras.losses.MeanAbsoluteError()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        
    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    #@tf.function
    @tf.autograph.experimental.do_not_convert
    def train_step(self, data):

        bev = data['bev']         
        #teacher = data['teacher']    # dict of (B,180,180,Ck)
        tokenss = data['tokenss']

        with tf.GradientTape() as tape:

            out = self.student(bev, training=True)
            loss = 0.0

            preds_dicts = raw_preds_2_preds_dicts(raw_preds=out, dimensions=dimensions)

            out_list = load_example_py(tokens=tokenss)
        
            example_keys = ['gt_boxes_and_cls', 'hm', 'anno_box', 'ind', 'mask', 'cat', 'gt_box']  # vagy ahány van
            ex = {k: out_list[i] for i, k in enumerate(example_keys)}

            # LABEL LOSS: --------------------------------------------------------------------------

            for task_id, preds_dict in enumerate(preds_dicts):

                # focal loss:
                preds_dict['hm'] = tf.sigmoid(preds_dict['hm'])
                loss = loss + focal_loss(
                    preds_dict['hm'],
                    ex['hm'][task_id],   
                    ex['ind'][task_id],
                    ex['mask'][task_id],
                    ex['cat'][task_id]
                )

                # reg loss:

                target_box = ex['anno_box'][task_id]
                # reconstruct the anno_box from multiple reg heads
                
                preds_dict['anno_box'] = tf.concat((preds_dict['reg'], preds_dict['height'], preds_dict['dim'],
                                                            preds_dict['vel'], preds_dict['rot']), axis=3)
                
                loss = loss + tf.reduce_sum(centernet_loss.RegLoss()(preds_dict['anno_box'], ex['mask'][task_id], ex['ind'][task_id], target_box) * code_weights)

                # iou loss:

                batch_dim = tf.exp(tf.clip_by_value(preds_dict['dim'], clip_value_min=-5.0, clip_value_max=5.0))
                batch_rot = preds_dict['rot']
                batch_rots = batch_rot[..., 0:1]
                batch_rotc = batch_rot[..., 1:2]
                batch_reg = preds_dict['reg']
                batch_hei = preds_dict['height']
                batch_rot = tf.atan2(batch_rots, batch_rotc)

                shape = tf.shape(batch_dim)
                batch = shape[0]
                H = shape[1]
                W = shape[2]

                batch_reg = tf.reshape(batch_reg, (batch, H * W, 2))
                batch_hei = tf.reshape(batch_hei, (batch, H * W, 1))
                batch_rot = tf.reshape(batch_rot, (batch, H * W, 1))
                batch_dim = tf.reshape(batch_dim, (batch, H * W, 3))


                ys, xs = tf.meshgrid(tf.range(H), tf.range(W), indexing='ij')
                ys = tf.tile(tf.expand_dims(ys, axis=0), [batch, 1, 1])
                xs = tf.tile(tf.expand_dims(xs, axis=0), [batch, 1, 1])

                # a dtype-ot igazítsd a batch_dim-hez
                ys = tf.cast(ys, batch_dim.dtype)
                xs = tf.cast(xs, batch_dim.dtype)

                xs = tf.reshape(xs, (batch, -1, 1)) + batch_reg[:, :, 0:1]
                ys = tf.reshape(ys, (batch, -1, 1)) + batch_reg[:, :, 1:2]

                xs = xs * 8 * 0.075 - 54
                ys = ys * 8 * 0.075 - 54
                batch_box_preds = tf.concat([xs, ys, batch_hei, batch_dim, batch_rot], axis=-1)
                batch_box_preds = tf.reshape(batch_box_preds, (batch, H, W, -1))

                loss = loss + centernet_loss.IouLoss()(preds_dict['iou'], ex['mask'][task_id], ex['ind'][task_id],
                                                    batch_box_preds, ex['gt_box'][task_id])
                
                # ioureg loss:

                loss = loss + 0.25*centernet_loss.IouRegLoss('DIoU')(batch_box_preds, ex['mask'][task_id], ex['ind'][task_id],
                                                     ex['gt_box'][task_id])
                
            # Teacher loss: ----------------------------------------------------------------------------------------

            # for pred_dict, teacher_dict in zip(preds_dicts, teacher_dicts):

            #     mask_fg = tf.cast(make_fg_mask_from_hm(teacher_dict['hm']), dtype=tf.float32)

            #     ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #             labels=tf.sigmoid(teacher_dict['hm']),  # teacher prob
            #             logits=pred_dict['hm']               # student logit
            #         )
                
            #     ce_loss = tf.reduce_sum(ce_loss * mask_fg) / (tf.reduce_sum(mask_fg) + 1e-6)
                
            #     loss = loss + self.lambda_2*ce_loss

            #     for name in ['reg', 'height','dim','rot','vel','iou']:

            #         mse_loss_per_pixel = tf.square(teacher_dict[name] - pred_dict[name])

            #         C = teacher_dict[name].shape[-1]
            #         mask_new = tf.tile(mask_fg, [1,1,1,C])
            #         weighted_mse_loss_per_pixel = mse_loss_per_pixel * mask_new

            #         head_mse_loss = tf.reduce_sum(weighted_mse_loss_per_pixel) / tf.reduce_sum(mask_fg)

            #         loss = loss + self.lambda_2 * self.w[name] * head_mse_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        self.train_loss.update_state(loss)
        
        return {"loss": self.train_loss.result()}
    
# Dataset

def make_streaming_dataset_pillars(tokens, point_folder, batch_size=16, shuffle=True):
    
    def gen():
        idxs = np.arange(len(tokens))
        if shuffle:
            np.random.shuffle(idxs)

        for s in range(0, len(tokens), batch_size):
            sel = idxs[s:s+batch_size]

            bevs, tokenss = [], []
            
            for j in sel:
                smp = tokens[j]

                with open(os.path.join(point_folder, smp+'.pkl'), 'rb') as f:

                    bev = pickle.load(f)

                bevs.append(torch_or_np_to_hwC(bev[smp]))

                tokenss.append(smp)


            # konvertálás RaggedTensor-ba közvetlenül
            batch = {
                "bev": tf.convert_to_tensor(np.stack(bevs, axis=0), dtype=tf.float32),
                "tokenss": tf.convert_to_tensor(tokenss, dtype=tf.string),
            }
            yield batch

    return gen