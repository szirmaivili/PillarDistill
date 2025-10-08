from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

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


def build_student_flatten_decoder_bigger(input_shape, heads):
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

    outputs = {
        'hm': layers.Conv2D(heads['hm'], 1, 1, padding='same', use_bias=True, name='conv2d_15')(x),
        'reg': layers.Conv2D(heads['reg'], 1, 1, padding='same', use_bias=True, name='conv2d_17')(x),
        'height': layers.Conv2D(heads['height'], 1, 1, padding='same', use_bias=True, name='conv2d_14')(x),
        'dim': layers.Conv2D(heads['dim'], 1, 1, padding='same', use_bias=True, name='conv2d_13')(x),
        'rot': layers.Conv2D(heads['rot'], 1, 1, padding='same', use_bias=True, name='conv2d_18')(x),
        'vel': layers.Conv2D(heads['vel'], 1, 1, padding='same', use_bias=True, name='conv2d_19')(x),
        'iou': layers.Conv2D(heads['iou'], 1, 1, padding='same', use_bias=True, name='conv2d_16')(x),
    }

    heads_model = Model(heads_input, outputs, name='student_bev_flatten_relu6_big_heads')
    full_model = Sequential([base_model, heads_model], name='student_bev_flatten_relu6_big')

    return full_model, base_model, heads_model


""" Example conversion (tested with random inputs)

heads = {'hm':10, 'reg': 12, 'height':6,'dim':18,'rot':12,'vel':12,'iou':6}
full, model, head = build_student_flatten_decoder_bigger((1440, 1440, 32), heads=heads)

import akida
from cnn2snn import set_akida_version, AkidaVersion
from quantizeml.models import quantize, QuantizationParams

qparams = QuantizationParams(
    input_dtype="int8",
    weight_bits=4,
    input_weight_bits=4,
    activation_bits=4,
    per_tensor_activations=True,
    output_bits=4,
)

with set_akida_version(AkidaVersion.v1):
    model_quantized = quantize(
        model,
        qparams=qparams,
        num_samples=1,
        batch_size=64,
        epochs=32,
    )

from cnn2snn import convert

with set_akida_version(AkidaVersion.v1):
    model_akida = convert(model_quantized)

devices = akida.devices()
model_akida.map(device=devices[0])

import numpy as np
random_input = np.random.rand(1, 1440,1440, 32)
random_input = np.clip(random_input, 0, 16).astype(np.uint8)
model_akida.forward(random_input)

"""