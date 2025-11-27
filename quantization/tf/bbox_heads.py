import tensorflow as tf
from tensorflow.keras import layers, models, initializers


# --- Conv + ReLU6 helper ---
def conv_bn_relu6(in_ch, out_ch, k=3, s=1, p=1):
    """TensorFlow megfelelője a PyTorch conv_bn_relu6-nek."""
    return models.Sequential([
        layers.Conv2D(
            out_ch,
            kernel_size=k,
            strides=s,
            padding='same' if p == 1 else 'valid',
            use_bias=False
        ),
        layers.ReLU(max_value=6)
    ])


# --- Egyetlen head (reg, height, dim, rot, vel, iou, hm) ---
def make_head(in_ch, out_ch, is_hm=False):
    head = models.Sequential([
        conv_bn_relu6(in_ch, 64),

        layers.Conv2D(
            out_ch,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=(
                initializers.RandomNormal(mean=0.0, stddev=0.001)
                if is_hm else initializers.HeNormal()
            ),
            bias_initializer=(
                tf.keras.initializers.Constant(-2.19) if is_hm
                else tf.keras.initializers.Zeros()
            )
        )
    ])

    return head


# === Egy task-hoz tartozó SepHead (7 külön head) ===
# class SepHead(tf.keras.Model):
#     def __init__(self, num_classes=2, hm_channels_override=None):
#         super(SepHead, self).__init__()

#         self.head_defs = {
#             "reg": 2,
#             "height": 1,
#             "dim": 3,
#             "rot": 2,
#             "vel": 2,
#             "iou": 1,
#             "hm": hm_channels_override if hm_channels_override is not None else num_classes,
#         }

#         # Létrehozzuk a 7 head-et
#         for head_name, out_ch in self.head_defs.items():
#             is_hm = (head_name == "hm")
#             setattr(self, head_name, make_head(64, out_ch, is_hm=is_hm))

    # def call(self, x, training=False):
    #     outputs = {}
    #     for head_name in self.head_defs.keys():
    #         outputs[head_name] = getattr(self, head_name)(x)
    #     return outputs

class SepHead(tf.keras.Model):
    """
    PyTorch-kulcs kompatibilis 7 fej / task.
    head_name: pl. 'reg', 'height', 'dim', 'rot', 'vel', 'iou', 'hm'
    task_id: 0..5
    """
    def __init__(self, in_ch, out_ch, head_name, task_id, is_hm=False):
        super(SepHead, self).__init__()

        # ---- First Conv ----
        self.conv0 = layers.Conv2D(
            64, kernel_size=3, strides=1, padding='same',
            use_bias=False,
            name=f"bbox_head_tasks_{task_id}_{head_name}_0_0"
        )

        # ---- ReLU ----
        self.relu0 = layers.ReLU(
            max_value=6,
            name=f"bbox_head_tasks_{task_id}_{head_name}_0_1"
        )

        # ---- Final Conv ----
        self.conv1 = layers.Conv2D(
            out_ch,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=(
                initializers.RandomNormal(mean=0.0, stddev=0.001)
                if is_hm else initializers.HeNormal()
            ),
            bias_initializer=(
                tf.keras.initializers.Constant(-2.19) if is_hm
                else tf.keras.initializers.Zeros()
            ),
            name=f"bbox_head_tasks_{task_id}_{head_name}_1"
        )

    def call(self, x, training=False):
        x = self.conv0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        return x

# === Fő Bbox modul (CenterHead megfelelője) ===
# class Bbox(tf.keras.Model):
#     def __init__(self, num_tasks=6):
#         super(Bbox, self).__init__()

#         # Shared convolution
#         self.shared_conv = models.Sequential([
#             layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False),
#             layers.ReLU(max_value=6),
#         ])

#         # Heatmap csatornaszám taskonként
#         hm_channels = [1, 2, 2, 1, 2, 2]

#         # 6 db SepHead
#         self.tasks = []
#         for i in range(num_tasks):
#             self.tasks.append(SepHead(num_classes=hm_channels[i]))

#         # osztálynevek → opcionális
#         self.class_names = [
#             ['car'],
#             ['truck', 'construction_vehicle'],
#             ['bus', 'trailer'],
#             ['barrier'],
#             ['motorcycle', 'bicycle'],
#             ['pedestrian', 'traffic_cone']
#         ]

#     def call(self, x, training=False):
#         x = self.shared_conv(x)

#         outputs = []
#         for task_head in self.tasks:
#             outputs.append(task_head(x))

#         return outputs

class Bbox(tf.keras.Model):
    def __init__(self, num_tasks=6):
        super(Bbox, self).__init__()

        # =========================================================
        # -------------------- SHARED CONV ------------------------
        # =========================================================
        self.shared_conv = models.Sequential([
            layers.Conv2D(
                64, kernel_size=3, strides=1, padding='same',
                use_bias=False,
                name="bbox_head_shared_conv_0"
            ),
            layers.ReLU(
                max_value=6,
                name="bbox_head_shared_conv_1"
            ),
        ], name="bbox_head_shared_conv")

        # =========================================================
        # ----------------------- TASK HEADS -----------------------
        # =========================================================

        # PyTorch kulcsok alapján:
        # hm channels taskonként (1,2,2,1,2,2)
        self.hm_channels = [1, 2, 2, 1, 2, 2]

        self.task_heads = []

        for task_id in range(num_tasks):

            heads = {}

            heads["reg"] =      SepHead(64, 2,   "reg",    task_id)
            heads["height"] =   SepHead(64, 1,   "height", task_id)
            heads["dim"] =      SepHead(64, 3,   "dim",    task_id)
            heads["rot"] =      SepHead(64, 2,   "rot",    task_id)
            heads["vel"] =      SepHead(64, 2,   "vel",    task_id)
            heads["iou"] =      SepHead(64, 1,   "iou",    task_id)
            heads["hm"] =       SepHead(64, self.hm_channels[task_id], "hm", task_id, is_hm=True)

            self.task_heads.append(heads)


    def call(self, x, training=False):
        x = self.shared_conv(x)

        outputs = []
        for task_id in range(len(self.task_heads)):
            out = {}
            for key, head in self.task_heads[task_id].items():
                out[key] = head(x)
            outputs.append(out)

        return outputs
