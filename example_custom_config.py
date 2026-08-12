from quark.torch import LLMTemplate

# Get a specific template
llama_template = LLMTemplate.get("llama")

# Create configuration with multiple options
config = llama_template.get_config(
    scheme="int4_wo_128",          # Global quantization scheme
    algorithm="awq",               # Quantization algorithm
    kv_cache_scheme="fp8",         # KV cache quantization
    min_kv_scale=1.0,              # Minimum value of KV Cache scale
    attention_scheme="fp8",        # Attention quantization
    layer_config={                 # Layer-specific configurations
        "*.mlp.gate_proj": "mxfp4",
        "*.mlp.up_proj": "mxfp4",
        "*.mlp.down_proj": "mxfp4"
    },
    layer_type_config={            # Layer type configurations
        nn.LayerNorm: "fp8"
    },
    exclude_layers=["lm_head"]      # Exclude layers from quantization
)

from quark.torch import LLMTemplate

# Create a new template
template = LLMTemplate(
  model_type="kimi_k2",
  kv_layer_name=["*kv_b_proj"],
  exclude_layers=["lm_head"]
)

# Register the template to LLMTemplate class (optional, if you want to use the template in other places)
LLMTemplate.register_template(template)

# Get the template
template = LLMTemplate.get("kimi_k2")

# Create a configuration
config = template.get_config(
    scheme="fp8",
    kv_cache_scheme="fp8"
)


from quark.torch.quantization.config.config import Int8PerTensorSpec, QLayerConfig
from quark.torch import LLMTemplate

# Create custom quantization specification
quant_spec = Int8PerTensorSpec(
    observer_method="min_max",
    symmetric=True,
    scale_type="float",
    round_method="half_even",
    is_dynamic=False
).to_quantization_spec()

# Create and register custom scheme
global_config = QLayerConfig(weight=quant_spec)
LLMTemplate.register_scheme("custom_int8_wo", config=global_config)

# Get a specific template
llama_template = LLMTemplate.get("llama")

# Use custom scheme
config = llama_template.get_config(scheme="custom_int8_wo")

import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization.config.config import QConfig, QTensorConfig, QLayerConfig, SmoothQuantConfig
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver

in_feat = 32 * 128
out_feat = 64 * 128

class MySubModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_feat, bias=False)
        self.lin1 = nn.Linear(in_feat, out_feat, bias=False)
        self.lin1.weight.data = torch.normal(0, 1, (out_feat, in_feat))

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.lin1(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        # We put the Linear + LayerNorm in a ModuleList, which is expected by AMD Quark,
        # as the implementation is tailored for multi-layer transformer models.
        self.layers = nn.ModuleList([MySubModule() for i in range(1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = MyModel()
model = model.eval()
model_copy = copy.deepcopy(model)

# Create reference tensor with long tail.
inp = torch.empty(1, in_feat)
inp.cauchy_(sigma=5e-3)
inp = inp + torch.normal(0, 1, (out_feat, in_feat))

# Save the reference output.
with torch.no_grad():
    res_orig = model(inp)

# Quantize the model using smoothquant.
quant_spec = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
    ch_axis=None,
    group_size=None
)
global_config = QLayerConfig(weight=quant_spec, input_tensors=quant_spec)
smoothquant_config = SmoothQuantConfig(
    scaling_layers=[{"prev_op": "layer_norm", "layers": ["lin1"], "inp": "lin1"}],
    model_decoder_layers="layers",
    alpha=0.5,
    scale_clamp_min=1e-12,
)
quant_config = QConfig(global_quant_config=global_config, algo_config=[smoothquant_config])

quantizer = ModelQuantizer(quant_config)
calib_dataloader = DataLoader([{"x": inp}])

quant_model_smooth = quantizer.quantize_model(model, calib_dataloader)
quant_model_smooth = quant_model_smooth.eval()

with torch.no_grad():
    res_quant_smooth = quant_model_smooth(inp)

# Quantize the model without using smoothquant.
quant_config = QConfig(global_quant_config=global_config)

quantizer = ModelQuantizer(quant_config)

quant_model_nonsmooth = quantizer.quantize_model(model_copy, calib_dataloader)
quant_model_nonsmooth = quant_model_nonsmooth.eval()

with torch.no_grad():
    res_quant_nonsmooth = quant_model_nonsmooth(inp)

print("L1 error non-smooth:", (res_orig - res_quant_nonsmooth).abs().mean())
print("L1 error smooth:", (res_orig - res_quant_smooth).abs().mean())