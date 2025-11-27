import tensorflow as tf
from tensorflow.keras import layers, models

class Neck(tf.keras.Model):
    def __init__(self, in_channels=192, mid_channels=160, out_channels=96):
        super(Neck, self).__init__()

        # --- block_5 ---
        block5_layers = [
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.Conv2D(mid_channels, kernel_size=3, strides=1, padding='valid', use_bias=False),
            layers.ReLU(max_value=6),
        ]

        for _ in range(5):
            block5_layers.extend([
                layers.Conv2D(mid_channels, kernel_size=3, strides=1, padding='same', use_bias=False),
                layers.ReLU(max_value=6),
            ])

        self.block_5 = models.Sequential(block5_layers)

        # --- deblock_5 ---
        self.deblock_5 = models.Sequential([
            layers.Conv2DTranspose(out_channels, kernel_size=2, strides=2, use_bias=False),
            layers.ReLU(max_value=6)
        ])

        # --- block_4 ---
        block4_layers = [
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.Conv2D(out_channels * 2, kernel_size=3, strides=1, padding='valid', use_bias=False),
            layers.ReLU(max_value=6),
        ]

        for _ in range(5):
            block4_layers.extend([
                layers.Conv2D(out_channels * 2, kernel_size=3, strides=1, padding='same', use_bias=False),
                layers.ReLU(max_value=6),
            ])

        self.block_4 = models.Sequential(block4_layers)

        # --- deblock_4 ---
        self.deblock_4 = models.Sequential([
            layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
            layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='valid', use_bias=False),
            layers.ReLU(max_value=6),
        ])

    def call(self, conv_4, conv_5, training=False):
        up_4 = self.deblock_4(conv_4)
        up_5 = self.deblock_5(self.block_5(conv_5))

        # PyTorch: cat(dim=1) = channel dim
        # TensorFlow: axis=-1
        x = tf.concat([up_4, up_5], axis=-1)

        x = self.block_4(x)
        return x
    
class Neck_mod(tf.keras.Model):
    def __init__(self, in_channels=192, mid_channels=160, out_channels=96):
        super(Neck_mod, self).__init__()

        # =========================================================
        # ------------------------ block_4 -------------------------
        # =========================================================

        block4_layers = [

            # 0: ZeroPadding (PyTorch-ban nincs megfelelő kulcs)
            layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="neck_block_4_0"),

            # 1: Conv2D
            layers.Conv2D(
                out_channels*2, kernel_size=3, strides=1, padding="valid",
                use_bias=False, name="neck_block_4_1"
            ),

            # 2: ReLU
            layers.ReLU(max_value=6, name="neck_block_4_2"),
        ]

        # ismétlődő 5 blokk: (Conv, ReLU)
        # PyTorch indexek: (3,4), (5,6), (7,8), (9,10), (11,12)

        block4_indices = [3,4,5,6,7,8,9,10,11,12]

        for idx in range(5):  # 5 blokk
            conv_idx = block4_indices[2*idx]
            relu_idx = block4_indices[2*idx + 1]

            block4_layers.append(
                layers.Conv2D(
                    out_channels*2, kernel_size=3, strides=1, padding="same",
                    use_bias=False, name=f"neck_block_4_{conv_idx}"
                )
            )
            block4_layers.append(
                layers.ReLU(max_value=6, name=f"neck_block_4_{relu_idx}")
            )

        self.block_4 = models.Sequential(block4_layers, name="neck_block_4")


        # =========================================================
        # ----------------------- deblock_4 ------------------------
        # =========================================================

        self.deblock_4 = models.Sequential([

            # 0 – ZeroPadding
            layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="neck_deblock_4_0"),

            # 1 – Conv2D
            layers.Conv2D(
                out_channels*2, kernel_size=3, strides=1, padding="valid",
                use_bias=False, name="neck_deblock_4_1"
            ),

            # 2 – ReLU
            layers.ReLU(max_value=6, name="neck_deblock_4_2"),

        ], name="neck_deblock_4")


    def call(self, conv_4, conv_5=None, training=False):
        up_4 = self.deblock_4(conv_4)
        x = self.block_4(up_4)
        return x

# class Neck_mod(tf.keras.Model):
#     def __init__(self, in_channels=192, mid_channels=160, out_channels=96):
#         super(Neck, self).__init__()

#         # --- block_4 ---
#         block4_layers = [
#             layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
#             layers.Conv2D(out_channels * 2, kernel_size=3, strides=1, padding='valid', use_bias=False),
#             layers.ReLU(max_value=6),
#         ]

#         for _ in range(5):
#             block4_layers.extend([
#                 layers.Conv2D(out_channels * 2, kernel_size=3, strides=1, padding='same', use_bias=False),
#                 layers.ReLU(max_value=6),
#             ])

#         self.block_4 = models.Sequential(block4_layers)

#         # --- deblock_4 ---
#         self.deblock_4 = models.Sequential([
#             layers.ZeroPadding2D(padding=((1, 1), (1, 1))),
#             layers.Conv2D(out_channels*2, kernel_size=3, strides=1, padding='valid', use_bias=False),
#             layers.ReLU(max_value=6),
#         ])

#     def call(self, conv_4, conv_5, training=False):
#         up_4 = self.deblock_4(conv_4)
        
#         x = self.block_4(up_4)
#         return x