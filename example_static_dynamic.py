# 1. Set model
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

# 2. Set quantization configuration
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization.config.config import QConfig, QTensorConfig, QLayerConfig
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver

# 2-1. For weight only quantization, please uncomment the following lines.
DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QTensorConfig(dtype=Dtype.uint4,
                                                  observer_cls=PerChannelMinMaxObserver,
                                                  symmetric=False,
                                                  scale_type=ScaleType.float,
                                                  round_method=RoundType.half_even,
                                                  qscheme=QSchemeType.per_group,
                                                  ch_axis=0,
                                                  is_dynamic=False,
                                                  group_size=32)
DEFAULT_W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)
quant_config = QConfig(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG)

# 2-2. For dynamic quantization, please uncomment the following lines.
# INT8_PER_TENSER_DYNAMIC_SPEC = QTensorConfig(dtype=Dtype.int8,
#                                              qscheme=QSchemeType.per_tensor,
#                                              observer_cls=PerTensorMinMaxObserver,
#                                              symmetric=True,
#                                              scale_type=ScaleType.float,
#                                              round_method=RoundType.half_even,
#                                              is_dynamic=True)
# DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QLayerConfig(input_tensors=INT8_PER_TENSER_DYNAMIC_SPEC,
#                                                                weight=INT8_PER_TENSER_DYNAMIC_SPEC)
# quant_config = QConfig(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG)

# 2-3. For static quantization , please uncomment the following lines.
# FP8_PER_TENSOR_SPEC = QTensorConfig(dtype=Dtype.fp8_e4m3,
#                                     qscheme=QSchemeType.per_tensor,
#                                     observer_cls=PerTensorMinMaxObserver,
#                                     is_dynamic=False)
# DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC,
#                                                     weight=FP8_PER_TENSOR_SPEC)
# quant_config = QConfig(global_quant_config=DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG)

# 3. Define calibration dataloader (still need this step for weight only and dynamic quantization)
from torch.utils.data import DataLoader
text = "Hello, how are you?"
tokenized_outputs = tokenizer(text, return_tensors="pt")
calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

# 4. In-place replacement with quantized modules in model
from quark.torch import ModelQuantizer
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

# from quark.torch.quantization import FP8E4M3PerTensorSpec
# from quark.torch.quantization.config.config import QConfig, QLayerConfig

# fp8_dyn = FP8E4M3PerTensorSpec(is_dynamic=True).to_quantization_spec()
# quant_config = QConfig(global_quant_config=QLayerConfig(weight=fp8_dyn, input_tensors=fp8_dyn))