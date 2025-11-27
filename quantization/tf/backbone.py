import tensorflow as tf
from tensorflow.keras import layers, models

class Backbone(tf.keras.Model):
    def __init__(self, input_channels=32):
        super(Backbone, self).__init__()

        # ------- conv1 -------
        self.conv1 = models.Sequential([
            # Block 1: 3×Conv2D + ReLU6
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(32, (3,3), strides=(1,1), padding='same', input_shape=(None, None, input_channels), name="backbone_conv1_0_0_0")
                ]),
                models.Sequential([
                    layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_0_1_0")
                ]),
                models.Sequential([
                    layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_0_2_0")
                ]),
                layers.ReLU(max_value=6)
            ], name="backbone_conv1_block0"),

            # Block 2: 2×Conv2D + ReLU6
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_1_0_0")
                ]),
                models.Sequential([
                    layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_1_1_0")
                ]),
                layers.ReLU(max_value=6)
            ], name = "backbone_conv1_block1")
        ], name = "backbone_conv1")

        # ------- conv2 -------
        self.conv2 = models.Sequential([
            layers.Conv2D(64, (3,3), strides=(2,2), padding='same', use_bias=False, name = "backbone_conv2_0"),
            layers.ReLU(max_value=6),

            # Block 1: 2×Conv2D + ReLU6
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_2_0_0")
                ]),
                models.Sequential([
                    layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_2_1_0")
                ]),
                layers.ReLU(max_value=6)
            ], name = "backbone_conv2_block0"),

            # Block 2: 2×Conv2D + ReLU6
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_3_0_0")
                ]),
                models.Sequential([
                    layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_3_1_0")
                ]),
                layers.ReLU(max_value=6)
            ], name= "backbone_conv2_block1")
        ], name = "backbone_conv2")

        # ------- conv3 -------
        self.conv3 = models.Sequential([
            layers.Conv2D(128, (3,3), strides=(2,2), padding='same', use_bias=False),
            layers.ReLU(max_value=6),

            # Block 1
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
                ]),
                models.Sequential([
                    layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
                ]),
                layers.ReLU(max_value=6)
            ]),

            # Block 2
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
                ]),
                models.Sequential([
                    layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
                ]),
                layers.ReLU(max_value=6)
            ])
        ])

        # ------- conv4 -------
        self.conv4 = models.Sequential([
            layers.Conv2D(192, (3,3), strides=(2,2), padding='same', use_bias=False),
            layers.ReLU(max_value=6),

            # Block 1
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
                ]),
                models.Sequential([
                    layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
                ]),
                layers.ReLU(max_value=6)
            ]),

            # Block 2
            models.Sequential([
                models.Sequential([
                    layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
                ]),
                models.Sequential([
                    layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
                ]),
                layers.ReLU(max_value=6)
            ])
        ])

        # ------- conv5 -------
        self.conv5 = models.Sequential([
            layers.Conv2D(192, (3,3), strides=(2,2), padding='same', use_bias=False),
            layers.ReLU(max_value=6),

            # Block 1
            models.Sequential([
                layers.Conv2D(192, (3,3), strides=(1,1), padding='same', use_bias=False),
                layers.ReLU(max_value=6)
            ]),

            # Block 2
            models.Sequential([
                layers.Conv2D(192, (3,3), strides=(1,1), padding='same', use_bias=False),
                layers.ReLU(max_value=6)
            ])
        ])

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        conv_4 = x
        x = self.conv5(x)
        conv_5 = x
        return conv_4, conv_5

# class Backbone_mod(tf.keras.Model):

#     def __init__(self, input_channels=32):
#         super(Backbone, self).__init__()

#         # ------- conv1 -------
#         self.conv1 = models.Sequential([
#             # Block 1: 3×Conv2D + ReLU6
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(32, (3,3), strides=(1,1), padding='same', input_shape=(None, None, input_channels), name="backbone_conv1_0_0_0")
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_0_1_0")
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_0_2_0")
#                 ]),
#                 layers.ReLU(max_value=6)
#             ], name="backbone_conv1_block0"),

#             # Block 2: 2×Conv2D + ReLU6
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_1_0_0")
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name="backbone_conv1_1_1_0")
#                 ]),
#                 layers.ReLU(max_value=6)
#             ], name = "backbone_conv1_block1")
#         ], name = "backbone_conv1")

#         # ------- conv2 -------
#         self.conv2 = models.Sequential([
#             layers.Conv2D(64, (3,3), strides=(2,2), padding='same', use_bias=False, name = "backbone_conv2_0"),
#             layers.ReLU(max_value=6),

#             # Block 1: 2×Conv2D + ReLU6
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_2_0_0")
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_2_1_0")
#                 ]),
#                 layers.ReLU(max_value=6)
#             ], name = "backbone_conv2_block0"),

#             # Block 2: 2×Conv2D + ReLU6
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_3_0_0")
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name = "backbone_conv2_3_1_0")
#                 ]),
#                 layers.ReLU(max_value=6)
#             ], name= "backbone_conv2_block1")
#         ], name = "backbone_conv2")

#         # ------- conv3 -------
#         self.conv3 = models.Sequential([
#             layers.Conv2D(128, (3,3), strides=(2,2), padding='same', use_bias=False),
#             layers.ReLU(max_value=6),

#             # Block 1
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 layers.ReLU(max_value=6)
#             ]),

#             # Block 2
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(128, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 layers.ReLU(max_value=6)
#             ])
#         ])

#         # ------- conv4 -------
#         self.conv4 = models.Sequential([
#             layers.Conv2D(192, (3,3), strides=(2,2), padding='same', use_bias=False),
#             layers.ReLU(max_value=6),

#             # Block 1
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 layers.ReLU(max_value=6)
#             ]),

#             # Block 2
#             models.Sequential([
#                 models.Sequential([
#                     layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 models.Sequential([
#                     layers.Conv2D(192, (3,3), strides=(1,1), padding='same')
#                 ]),
#                 layers.ReLU(max_value=6)
#             ])
#         ])


#     def call(self, x, training=False):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         conv_4 = x
#         return conv_4

class Backbone_mod(tf.keras.Model):
    def __init__(self, input_channels=32):
        super(Backbone_mod, self).__init__()

        # =========================
        # ------- conv1 -------
        # =========================
        self.conv1 = models.Sequential([
            # Block 0: 3×Conv2D + ReLU6
            models.Sequential([
                layers.Conv2D(
                    32, (3,3), strides=1, padding='same',
                    input_shape=(None, None, input_channels),
                    name="backbone_conv1_0_0_0"
                ),
                layers.Conv2D(
                    32, (3,3), strides=1, padding='same',
                    name="backbone_conv1_0_1_0"
                ),
                layers.Conv2D(
                    32, (3,3), strides=1, padding='same',
                    name="backbone_conv1_0_2_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv1_0_3")
            ], name="backbone_conv1_block0"),

            # Block 1: 2×Conv2D + ReLU6
            models.Sequential([
                layers.Conv2D(
                    32, (3,3), strides=1, padding='same',
                    name="backbone_conv1_1_0_0"
                ),
                layers.Conv2D(
                    32, (3,3), strides=1, padding='same',
                    name="backbone_conv1_1_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv1_1_2")
            ], name="backbone_conv1_block1")
        ], name="backbone_conv1")


        # =========================
        # ------- conv2 -------
        # =========================
        self.conv2 = models.Sequential([

            layers.Conv2D(
                64, (3,3), strides=2, padding='same',
                use_bias=False, name="backbone_conv2_0"
            ),
            layers.ReLU(max_value=6, name="backbone_conv2_1"),

            # Block 0
            models.Sequential([
                layers.Conv2D(
                    64, (3,3), strides=1, padding='same',
                    name="backbone_conv2_2_0_0"
                ),
                layers.Conv2D(
                    64, (3,3), strides=1, padding='same',
                    name="backbone_conv2_2_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv2_2_2")
            ], name="backbone_conv2_block0"),

            # Block 1
            models.Sequential([
                layers.Conv2D(
                    64, (3,3), strides=1, padding='same',
                    name="backbone_conv2_3_0_0"
                ),
                layers.Conv2D(
                    64, (3,3), strides=1, padding='same',
                    name="backbone_conv2_3_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv2_3_2")
            ], name="backbone_conv2_block1")

        ], name="backbone_conv2")


        # =========================
        # ------- conv3 -------
        # =========================
        self.conv3 = models.Sequential([

            layers.Conv2D(
                128, (3,3), strides=2, padding='same',
                use_bias=False, name="backbone_conv3_0"
            ),
            layers.ReLU(max_value=6, name="backbone_conv3_1"),

            # Block 0
            models.Sequential([
                layers.Conv2D(
                    128, (3,3), strides=1, padding='same',
                    name="backbone_conv3_2_0_0"
                ),
                layers.Conv2D(
                    128, (3,3), strides=1, padding='same',
                    name="backbone_conv3_2_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv3_2_2")
            ], name="backbone_conv3_block0"),

            # Block 1
            models.Sequential([
                layers.Conv2D(
                    128, (3,3), strides=1, padding='same',
                    name="backbone_conv3_3_0_0"
                ),
                layers.Conv2D(
                    128, (3,3), strides=1, padding='same',
                    name="backbone_conv3_3_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv3_3_2")
            ], name="backbone_conv3_block1")

        ], name="backbone_conv3")


        # =========================
        # ------- conv4 -------
        # =========================
        self.conv4 = models.Sequential([

            layers.Conv2D(
                192, (3,3), strides=2, padding='same',
                use_bias=False, name="backbone_conv4_0"
            ),
            layers.ReLU(max_value=6, name="backbone_conv4_1"),

            # Block 0
            models.Sequential([
                layers.Conv2D(
                    192, (3,3), strides=1, padding='same',
                    name="backbone_conv4_2_0_0"
                ),
                layers.Conv2D(
                    192, (3,3), strides=1, padding='same',
                    name="backbone_conv4_2_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv4_2_2")
            ], name="backbone_conv4_block0"),

            # Block 1
            models.Sequential([
                layers.Conv2D(
                    192, (3,3), strides=1, padding='same',
                    name="backbone_conv4_3_0_0"
                ),
                layers.Conv2D(
                    192, (3,3), strides=1, padding='same',
                    name="backbone_conv4_3_1_0"
                ),
                layers.ReLU(max_value=6, name="backbone_conv4_3_2")
            ], name="backbone_conv4_block1")

        ], name="backbone_conv4")

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


