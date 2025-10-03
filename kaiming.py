import tensorflow as tf

def kaiming_init_tf(layer, a=0, mode="fan_out", nonlinearity="relu",
                    bias=0, distribution="normal"):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        initializer = tf.keras.initializers.HeUniform()
    else:
        initializer = tf.keras.initializers.HeNormal()
    # súly inicializálás
    if hasattr(layer, "kernel"):
        layer.kernel.assign(initializer(shape=layer.kernel.shape))
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.assign(tf.constant(bias, shape=layer.bias.shape))