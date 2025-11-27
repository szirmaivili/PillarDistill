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

def load_pytorch_state_dict(path):
    ckpt = torch.load(path, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    filtered = {}

    for k, v in sd.items():

        # --- Skip observer keys (not real weights) ---
        if any(x in k for x in [
            "act_min", "act_max",
            "weight_min", "weight_max",
            "tracked_min", "tracked_max",
            "running_mean", "running_var",
            "num_batches_tracked",
            "layer.weight", "layer.bias"
        ]):
            continue

        # Only keep real kernel / bias
        if k.endswith("kernel") or k.endswith("bias"):
            filtered[k] = v

    return filtered


# =============================================================
# 2) TF weight -> PyTorch key normalizáló
# =============================================================

def normalize_tf_weight_name(tf_name):
    """
    TF például ezt adja:
        'sep_head_21/bbox_head_tasks_3_reg_1/kernel:0'
    Ebből nekünk csak az utolsó komponens kell:
        'bbox_head_tasks_3_reg_1/kernel:0'
    """
    return tf_name.split("/")[-1]


def tf_to_pytorch_key(tf_name):
    """
    Átalakítjuk:
       bbox_head_tasks_3_reg_1/kernel:0
    →  bbox_head.tasks.3.reg.1.kernel
    """
    tf_name = tf_name.replace(":0", "")       # remove tensor index
    tf_name = tf_name.replace("/", ".")       # / → .
    tf_name = tf_name.replace("_", ".")       # _ → .

    parts = tf_name.split(".")
    # All parts except last are path, last is kernel/bias
    pt_key = ".".join(parts)
    return pt_key


# =============================================================
# 3) PT → TF súly transzponálás (Conv2D)
# =============================================================

def convert_conv_weight(pt_w):
    # PyTorch shape: (out, in, H, W)
    # TF shape:      (H, W, in, out)
    return np.transpose(pt_w, (2, 3, 1, 0))


# =============================================================
# 4) Teljes weight loader
# =============================================================

def load_weights_pt_to_tf(model, pt_state_dict):
    tf_vars = model.weights

    print("\n========== WEIGHT LOADING ==========\n")

    matched = 0
    missing_pt = []
    missing_tf = []

    # 1) Normalizált TF nevek
    tf_map = {}
    for var in tf_vars:
        tf_clean = normalize_tf_weight_name(var.name)
        pt_key = tf_to_pytorch_key(tf_clean)
        tf_map[pt_key] = var   # mapping: PT_key_form -> TF variable
        print(tf_clean)


    assert 0 == 2

    # 2) PyTorch -> TensorFlow betöltés
    for pt_key, pt_tensor in pt_state_dict.items():

        if pt_key not in tf_map:
            missing_tf.append(pt_key)
            continue

        tf_var = tf_map[pt_key]

        arr = pt_tensor.cpu().numpy()

        # Conv2D kernel transzponálása
        if len(tf_var.shape) == 4:
            arr = convert_conv_weight(arr)

        # Shape check
        if tuple(arr.shape) != tuple(tf_var.shape):
            print(f"[SHAPE MISMATCH] PT {pt_key}: {arr.shape}  TF {tf_var.shape}")
            continue

        tf_var.assign(arr)
        print(f"[OK] {pt_key:45s} --> {tf_var.name}")
        matched += 1

    # 3) Olyan TF súlyok listája, amikhez nem volt PT partner
    pt_keys_set = set(pt_state_dict.keys())
    tf_keys_set = set(tf_map.keys())

    missing_pt = list(tf_keys_set - pt_keys_set)  # these TF weights are unmatched

    print("\n========== SUMMARY ==========")
    print(f"Matched weights: {matched}")
    print(f"Missing PT weights: {len(missing_pt)}")
    print(f"Missing TF weights: {len(missing_tf)}")
    print("====================================\n")

    return missing_pt, missing_tf
    
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

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            # minden _load_from_bytes hívás CPU-ra kerül
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def cpu_load(path):
    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()

def main():

    dummy_input = tf.random.normal([1, 1440, 1440, 32])

    _ = model(dummy_input)

    print("Successful execution!")

    pt_state_dict = load_pytorch_state_dict(r'/PillarNet/PillarNet/quantization_wo_deconv/epoch_1.pth')

    # 3) Betöltés
    missing_pt, missing_tf = load_weights_pt_to_tf(model, pt_state_dict)

    assert 0 == 1

    pytorch_checkpoint = r'/PillarNet/PillarNet/quantization_wo_deconv/epoch_1.pth'
    tensorflow_save   = r'converted_tf_model'

    print("Loading PyTorch checkpoint...")
    state_dict = load_pytorch_state_dict(pytorch_checkpoint)

    # if 'state_dict' in ckpt:
    #     state_dict = ckpt['state_dict']
    # else:
    #     state_dict = ckpt

    pt_keys = list(state_dict.keys())

    print("Collecting TensorFlow weights...")
    tf_vars = model.weights
    print(f"TensorFlow model has {len(tf_vars)} variables.")

    def convert_conv_weight(pt_w):
        # PyTorch: (out, in, H, W)
        # TF     : (H, W, in, out)
        return np.transpose(pt_w, (2, 3, 1, 0))
    
    def assign_weight(tf_var, pt_tensor):
        tf_shape = tf_var.shape
        pt_array = pt_tensor.numpy()

        # Conv2D kernel
        if len(tf_shape) == 4:
            pt_array = convert_conv_weight(pt_array)

        tf_var.assign(pt_array)

    print("Converting weights...")

    pt_index = 0

    for tf_var in tf_vars:
        tf_shape = tuple(tf_var.shape)
        pt_name = pt_keys[pt_index]
        pt_tensor = state_dict[pt_name]

        pt_shape = tuple(pt_tensor.shape)

        # Ellenőrzés
        if len(tf_shape) == 4:
            # Conv2D kernel shape: TF: (H,W,in,out)
            # PyTorch: (out,in,H,W)
            if tf_shape != (pt_shape[2], pt_shape[3], pt_shape[1], pt_shape[0]):
                print(f"[WARNING] Shape mismatch Conv2D: TF {tf_shape} vs PT {pt_shape}")
        else:
            if tf_shape != pt_shape:
                print(f"[WARNING] Shape mismatch: TF {tf_shape} vs PT {pt_shape}")

        # Súly átmásolása
        assign_weight(tf_var, pt_tensor)

        print(f"[OK] {pt_name:50s} → {tf_var.name:40s}  {tf_shape}")

        pt_index += 1

    print("All weights converted successfully!")

    assert 0 == 2

    # model.save_weights(r'/PillarDistill/dense_pillarnet_tf')
    # print("Weights saved!")

    model.load_weights(r'/PillarDistill/dense_pillarnet_tf')
    print("Weights loaded!")

    inputs = cpu_load(r'/PillarNet/PillarNet/reader_output.pkl')

    final_input = []

    for input in inputs:

        arr = input.numpy()
        transposed_arr = np.transpose(arr, (0, 2, 3, 1))

        i = tf.convert_to_tensor(transposed_arr)

        final_input.append(i)

    outputs = []

    for Input in final_input:

        out = model(Input)
        outputs.append(out)

    with open(r'/PillarDistill/dense_pillarnet_outputs.pkl', 'wb') as f:

        pickle.dump(outputs, f)

    print("Outputs saved!")

if __name__ == "__main__":

    main()