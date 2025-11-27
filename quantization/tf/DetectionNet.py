import tensorflow as tf
from tensorflow.keras import Model
from backbone import Backbone_mod
from neck import Neck_mod
from bbox_heads import Bbox
import torch
import numpy as np
import pickle
import io
import tensorflow.keras as keras


class DetectionNet(Model):
    def __init__(self, backbone, neck, bbox_heads):
        """
        backbone: tf.keras.Model (pl. Backbone)
        neck: tf.keras.Model (pl. Neck)
        bbox_heads: tf.keras.Model (pl. Bbox)
        """
        super(DetectionNet, self).__init__()

        self.backbone = backbone
        self.neck = neck
        self.bbox_heads = bbox_heads

    def call(self, x, training=False):
        """
        Pipeline:
            x → backbone → (conv_4, conv_5)
                       → neck → fpn_features
                       → bbox_heads → detections
        """

        conv_4 = self.backbone(x, training=training)
        neck_out = self.neck(conv_4, training=training)
        outputs = self.bbox_heads(neck_out, training=training)

        return outputs

# Custom 4-bit Quantization Implementation (updated for STE gradients)
# We'll implement fake quantization with Straight-Through Estimator (STE)

class QuantizationAwareLayer(keras.layers.Layer):
    """
    Wrapper layer that applies 4-bit quantization-aware training.
    Uses STE for quantize-dequantize so gradients flow to original weights.
    """
    def __init__(self, layer, **kwargs):
        super(QuantizationAwareLayer, self).__init__(**kwargs)
        self.layer = layer
        self.num_bits = 4
        self.num_levels = 2 ** self.num_bits  # 16 levels for 4-bit
        
    def build(self, input_shape):
        self.layer.build(input_shape)
        # If the inner layer has weights, keep references to them
        self.kernel = None
        self.bias = None
        for w in self.layer.weights:
            if 'kernel' in w.name:
                self.kernel = w
            if 'bias' in w.name:
                self.bias = w

        # Create quantization range tracking variables for weights
        if self.kernel is not None:
            self.weight_min = self.add_weight(
                name=self.name + '_weight_min',
                shape=(),
                initializer=keras.initializers.Constant(-6.0),
                trainable=False
            )
            self.weight_max = self.add_weight(
                name=self.name + '_weight_max',
                shape=(),
                initializer=keras.initializers.Constant(6.0),
                trainable=False
            )
        
        # Create quantization range tracking variables for activations
        self.act_min = self.add_weight(
            name=self.name + '_act_min',
            shape=(),
            initializer=keras.initializers.Constant(0.0),
            trainable=False
        )
        self.act_max = self.add_weight(
            name=self.name + '_act_max',
            shape=(),
            initializer=keras.initializers.Constant(15.0),
            trainable=False
        )
        
        super(QuantizationAwareLayer, self).build(input_shape)
    
    def _fake_quantize_ste(self, x, min_val, max_val, symmetric=False):
        """Fake quantization using STE: quantize/dequantize in forward,
        but pass gradients through as identity (straight-through).
        """
        if symmetric:
            abs_max = tf.maximum(tf.abs(min_val), tf.abs(max_val))
            scale = abs_max / (self.num_levels / 2 - 1)
            # avoid division by zero
            scale = tf.where(scale == 0, tf.constant(1e-8, dtype=scale.dtype), scale)
            q = tf.round(x / scale)
            q = tf.clip_by_value(q, -(self.num_levels // 2), (self.num_levels // 2 - 1))
            dq = q * scale
        else:
            scale = (max_val - min_val) / (self.num_levels - 1)
            scale = tf.where(scale == 0, tf.constant(1e-8, dtype=scale.dtype), scale)
            q = tf.round((x - min_val) / scale)
            q = tf.clip_by_value(q, 0, self.num_levels - 1)
            dq = q * scale + min_val

        # STE: forward uses dequantized, backward passes gradient as if identity
        return x + tf.stop_gradient(dq - x)
    
    def call(self, inputs, training=None):
        # Update activation ranges in training mode
        if training:
            batch_min = tf.reduce_min(inputs)
            batch_max = tf.reduce_max(inputs)
            self.act_min.assign(0.9 * self.act_min + 0.1 * batch_min)
            self.act_max.assign(0.9 * self.act_max + 0.1 * batch_max)

        # Fake-quantize inputs (STE)
        q_inputs = self._fake_quantize_ste(inputs, self.act_min, self.act_max, symmetric=False)

        # If there are weights, fake-quantize them via STE but keep original vars for gradients
        if self.kernel is not None:
            if training:
                k_min = tf.reduce_min(self.kernel)
                k_max = tf.reduce_max(self.kernel)
                self.weight_min.assign(0.9 * self.weight_min + 0.1 * k_min)
                self.weight_max.assign(0.9 * self.weight_max + 0.1 * k_max)
            q_kernel = self._fake_quantize_ste(self.kernel, self.weight_min, self.weight_max, symmetric=True)
            # Compute Dense output manually to avoid reassigning kernel attr and to keep gradient path
            outputs = tf.linalg.matmul(q_inputs, q_kernel)
            if self.bias is not None:
                outputs = tf.nn.bias_add(outputs, self.bias)
            # Apply activation if inner layer has activation or is a ReLU wrapper
            if hasattr(self.layer, 'activation') and callable(self.layer.activation):
                outputs = self.layer.activation(outputs)
        else:
            # For non-weight layers (e.g., ReLU wrapper), just call the inner layer on quantized inputs
            outputs = self.layer(q_inputs, training=training)

        return outputs
    
    def get_config(self):
        config = super(QuantizationAwareLayer, self).get_config()
        config.update({
            'layer': keras.layers.serialize(self.layer),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        layer_config = config.pop('layer')
        layer = keras.layers.deserialize(layer_config)
        return cls(layer, **config)


def make_quantization_aware(model):
    """
    Convert a Keras model to use 4-bit quantization-aware training.
    Wraps Dense and activation layers with quantization.
    """
    def quantize_layer(layer):
        # Quantize Dense layers and ReLU activations
        if isinstance(layer, (keras.layers.Dense, keras.layers.ReLU)):
            return QuantizationAwareLayer(layer)
        return layer
    
    # Clone model with quantization wrappers
    quantized_layers = []
    for layer in model.layers:
        quantized_layers.append(quantize_layer(layer))
    
    quantized_model = keras.Sequential(quantized_layers)
    return quantized_model


print("Updated QuantizationAwareLayer defined (with STE).")
print(f"Quantization: {2**4} levels (4-bit) for weights and activations")

backbone = Backbone_mod(input_channels=32)
neck = Neck_mod()
bbox = Bbox(num_tasks=6)

backbone = make_quantization_aware(backbone)
neck = make_quantization_aware(neck)
bbox = make_quantization_aware(bbox)

model = DetectionNet(backbone, neck, bbox)

def main():

    dummy_input = tf.random.normal([1, 1440, 1440, 32])

    _ = model(dummy_input)

    print("Successful execution!")

if __name__ == "__main__":

    main()
