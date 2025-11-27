import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantizationAwareLayer(nn.Module):
    """
    Wrapper module that applies 4-bit quantization-aware training.
    Uses STE for quantize-dequantize so gradients flow to original weights.
    """
    def __init__(self, layer):
        super(QuantizationAwareLayer, self).__init__()
        self.layer = layer
        self.num_bits = 4
        self.num_levels = 2 ** self.num_bits  # 16 levels for 4-bit
        
        # In PyTorch, weights are usually accessed directly from the layer
        # Check if the wrapped layer has 'weight' (for Linear/Conv) and 'bias'
        self.kernel = None # This will store a reference to self.layer.weight if it exists
        self.bias = None   # This will store a reference to self.layer.bias if it exists

        if hasattr(self.layer, 'weight') and self.layer.weight is not None:
            self.kernel = self.layer.weight
        if hasattr(self.layer, 'bias') and self.layer.bias is not None:
            self.bias = self.layer.bias

        # Create quantization range tracking buffers for weights
        # These are nn.Module buffers, so they are saved with state_dict but are not trainable parameters
        if self.kernel is not None:
            self.register_buffer('weight_min', torch.tensor(-6.0, dtype=torch.float32))
            self.register_buffer('weight_max', torch.tensor(6.0, dtype=torch.float32))
        
        # Create quantization range tracking buffers for activations
        self.register_buffer('act_min', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('act_max', torch.tensor(15.0, dtype=torch.float32))
        
        # Momentum for EMA updates (0.1 means 10% new value, 90% old value)
        self.momentum = 0.01 

    def _fake_quantize_ste(self, x, min_val, max_val, symmetric=False):
        """
        Fake quantization using STE: quantize/dequantize in forward,
        but pass gradients through as identity (straight-through).
        """
        # Ensure min_val and max_val are detached to prevent gradient flow through them
        # (This is especially important if they were nn.Parameter, but here they are buffers)
        min_val_d = min_val.detach()
        max_val_d = max_val.detach()

        # Avoid division by zero
        epsilon = 1e-8

        epsilon_tensor = x.new_tensor(epsilon, dtype=x.dtype)
        
        if symmetric:
            # For 4-bit signed, range is [-8, 7]
            qmax_int = (self.num_levels // 2) - 1 # 7
            qmin_int = -(self.num_levels // 2)   # -8
            
            abs_max = torch.max(torch.abs(min_val_d), torch.abs(max_val_d))
            
            # Scale calculation
            scale_candidate = abs_max / qmax_int if qmax_int != 0 else x.new_tensor(1.0)
            
            # Biztosítjuk, hogy a scale sose legyen kisebb, mint epsilon_tensor.
            scale = torch.max(scale_candidate, epsilon_tensor)
            
            # Quantize to integer space
            q = torch.round(x / scale)
            
            # Clip to integer range
            q = torch.clamp(q, qmin_int, qmax_int)
            
            # Dequantize back to float
            dq = q * scale
        else: # Asymmetric
            # For 4-bit unsigned, range is [0, 15]
            qmax_int = self.num_levels - 1 # 15
            qmin_int = 0             # 0

            active_range = max_val_d - min_val_d
            
            scale_candidate = active_range / qmax_int if qmax_int != 0 else x.new_tensor(1.0)
            
            # Biztosítjuk, hogy a scale sose legyen kisebb, mint epsilon_tensor.
            scale = torch.max(scale_candidate, epsilon_tensor)
            
            # Quantize to integer space
            # Note: Keras uses (x - min_val) / scale. PyTorch needs x - min_val_d
            q = torch.round((x - min_val_d) / scale)
            
            # Clip to integer range
            q = torch.clamp(q, qmin_int, qmax_int)
            
            # Dequantize back to float
            # Note: Keras uses q * scale + min_val. PyTorch needs q * scale + min_val_d
            dq = q * scale + min_val_d

        # STE: forward uses dequantized, backward passes gradient as if identity
        return x + (dq - x).detach()
    
    def forward(self, inputs):
        # Determine if we are in training mode
        # self.training is a property of nn.Module, automatically set by .train() and .eval()
        training = self.training 

        # Update activation ranges in training mode using Exponential Moving Average (EMA)
        if training:
            # Detach to prevent gradient flow from these min/max calculations
            batch_min = torch.min(inputs).detach()
            batch_max = torch.max(inputs).detach()
            
            # Update buffers with EMA. Use torch.no_grad() or directly assign to .data
            # to prevent these updates from creating graph nodes.
            with torch.no_grad():
                self.act_min.mul_(1.0 - self.momentum).add_(batch_min, alpha=self.momentum)
                self.act_max.mul_(1.0 - self.momentum).add_(batch_max, alpha=self.momentum)
        
        # Fake-quantize inputs (STE)
        # Keras code uses symmetric=False for activations
        q_inputs = self._fake_quantize_ste(inputs, self.act_min, self.act_max, 
                                            symmetric=False)

        # If there are weights (e.g., Conv2d, Linear), fake-quantize them via STE
        outputs = None
        if self.kernel is not None:
            if training:
                # Update weight ranges in training mode using EMA
                k_min = torch.min(self.kernel).detach()
                k_max = torch.max(self.kernel).detach()
                
                with torch.no_grad():
                    self.weight_min.mul_(1.0 - self.momentum).add_(k_min, alpha=self.momentum)
                    self.weight_max.mul_(1.0 - self.momentum).add_(k_max, alpha=self.momentum)
            
            # Fake-quantize kernel (weights)
            # Keras code uses symmetric=True for weights
            q_kernel = self._fake_quantize_ste(self.kernel, self.weight_min, self.weight_max, 
                                                symmetric=True)
            
            # --- Perform the inner layer's operation using the quantized inputs and weights ---
            if isinstance(self.layer, nn.Linear):
                outputs = F.linear(q_inputs, q_kernel, self.bias)
            elif isinstance(self.layer, nn.Conv2d):
                outputs = F.conv2d(q_inputs, q_kernel, self.bias, 
                                   self.layer.stride, self.layer.padding, 
                                   self.layer.dilation, self.layer.groups)
                
            elif isinstance(self.layer, nn.ConvTranspose2d): # <-- ÚJ RÉSZ
                outputs = F.conv_transpose2d(q_inputs, q_kernel, self.bias,
                                             self.layer.stride, self.layer.padding,
                                             self.layer.output_padding, self.layer.groups,
                                             self.layer.dilation)
            else:
                # Fallback for layers with weights that are not Linear or Conv2d
                # This case might need specific handling or a different approach
                raise NotImplementedError(f"Wrapped layer type {type(self.layer)} with weights not yet supported for direct quantization operations in wrapper.")
            
            # The Keras example applies activation if inner layer has one.
            # In PyTorch, activations are usually separate layers.
            # If `self.layer` itself is an activation (e.g., nn.ReLU), it will be handled by the 'else' branch below.
            # If we wrapped a Keras-like Dense(activation='relu'), we'd apply it here.
            # For standard PyTorch models, activations are distinct layers.
        else: # For non-weight layers (e.g., nn.ReLU, nn.MaxPool2d, nn.Flatten)
            # Just call the inner layer's forward method on the quantized inputs
            outputs = self.layer(q_inputs)
        
        return outputs

# --- make_quantization_aware_pytorch function ---
def make_quantization_aware_pytorch(model, layer_types_to_quantize=(nn.Linear, nn.Conv2d, nn.ReLU, nn.ConvTranspose2d, nn.ReLU6)):
    """
    Convert a PyTorch model to use 4-bit quantization-aware training.
    Wraps specified layer types with QuantizationAwareLayer.
    """
    
    # Iterate through all named modules and replace them with QuantizationAwareLayer where applicable
    for name, module in model.named_children():
        # If the module is already a QuantizationAwareLayer, don't re-wrap it
        if isinstance(module, QuantizationAwareLayer):
            continue

        # If the module is a container (e.g., nn.Sequential, nn.ModuleList), recurse into it
        if len(list(module.children())) > 0:
            make_quantization_aware_pytorch(module, layer_types_to_quantize)
        
        # If it's a leaf module of a type we want to quantize
        elif isinstance(module, layer_types_to_quantize):
            # Replace the original module with the QuantizationAwareLayer wrapper
            # We need to get the parent module to set its attribute
            # This requires a bit more logic if the model is not a simple Sequential.
            # For a general solution, we'd need to keep track of parent modules.
            
            # A more robust way: use model._modules to access children
            setattr(model, name, QuantizationAwareLayer(module))
            
    return model


# --- Teszt modell és átalakítás ---
class SimplePyTorchModel(nn.Module):
    def __init__(self):
        super(SimplePyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

print("Original PyTorch model:")
original_model = SimplePyTorchModel()
print(original_model)

print("\nConverting to Quantization-Aware PyTorch model...")
# Clone the model first if you want to keep the original for comparison
qat_model = SimplePyTorchModel()
qat_model.load_state_dict(original_model.state_dict()) # Copy weights
qat_model = make_quantization_aware_pytorch(qat_model)
print(qat_model)

# --- Példa kalibrációra és futtatásra (EMA frissítés ellenőrzése) ---
print("\nRunning a dummy forward pass to update EMA ranges (training mode)...")
dummy_input = torch.randn(1, 1, 28, 28)
qat_model.train() # Set model to training mode for EMA updates

_ = qat_model(dummy_input)

# Check some updated ranges
print("\nUpdated range values after dummy pass:")
for name, module in qat_model.named_modules():
    if isinstance(module, QuantizationAwareLayer):
        print(f"Layer: {name} (wrapped {type(module.layer).__name__})")
        print(f"  Initial act_min/max: (0.0, 15.0)") # These are fixed initial values
        print(f"  Current act_min: {module.act_min.item():.4f}, act_max: {module.act_max.item():.4f}")
        if module.kernel is not None:
            print(f"  Initial weight_min/max: (-6.0, 6.0)") # Fixed initial values
            print(f"  Current weight_min: {module.weight_min.item():.4f}, weight_max: {module.weight_max.item():.4f}")

# Switch to eval mode (EMA updates stop)
qat_model.eval()
print("\nRunning a dummy forward pass (eval mode)...")
# Run a few times to simulate inference
for _ in range(3):
    _ = qat_model(dummy_input)

# Check values again, they should be the same as after the training pass
print("\nRange values after dummy pass (eval mode):")
for name, module in qat_model.named_modules():
    if isinstance(module, QuantizationAwareLayer):
        print(f"Layer: {name} (wrapped {type(module.layer).__name__})")
        print(f"  act_min: {module.act_min.item():.4f}, act_max: {module.act_max.item():.4f}")
        if module.kernel is not None:
             print(f"  weight_min: {module.weight_min.item():.4f}, weight_max: {module.weight_max.item():.4f}")