from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM
original_model_name="Qwen/Qwen3.6-27B"
model = AutoModelForImageTextToText.from_pretrained(original_model_name, torch_dtype="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(original_model_name)

from torch.utils.data import DataLoader
text = "Hello, how are you?"
tokenized_outputs = tokenizer(text, return_tensors="pt")
calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

from quark.torch import LLMTemplate
template = LLMTemplate(
    model_type="qwen3_5",
    q_layer_name=["*q_proj"],
    kv_layers_name=["*k_proj", "*v_proj"],
    exclude_layers_name=["*lm_head"]
)

LLMTemplate.register_template(template)
template = LLMTemplate.get("qwen3")
quant_config = template.get_config(scheme="ptpc_fp8", kv_cache_scheme="fp8")

# from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
# from quark.torch.quantization.config.config import QConfig, QTensorConfig, QLayerConfig
# from quark.torch.quantization.observer.observer import PerBlockBFPObserver
# DEFAULT_BFP16_PER_BLOCK = QTensorConfig(dtype=Dtype.int8,
#                                         symmetric=True,
#                                         observer_cls=PerBlockBFPObserver,
#                                         qscheme=QSchemeType.per_group, 
#                                         is_dynamic=False, 
#                                         ch_axis=-1,
#                                         scale_type=ScaleType.float,
#                                         group_size=8,
#                                         round_method=RoundType.half_even)

# DEFAULT_W_BFP16_PER_BLOCK_CONFIG = QLayerConfig(weight=DEFAULT_BFP16_PER_BLOCK)
# quant_config = QConfig(global_quant_config=DEFAULT_W_BFP16_PER_BLOCK_CONFIG)

from quark.torch import ModelQuantizer
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

from quark.torch import export_safetensors
export_model_name="./Qwen3.6-27B-ptpc_fp8"
export_safetensors(
    model=quant_model,
    output_dir=export_model_name
)
tokenizer.save_pretrained(export_model_name)

# from quark.torch import export_gguf

# model_dir = "meta-llama/Llama-2-7b-chat-hf"
# export_gguf(quantized_model, output_dir="./output_dir", model_type="llama", tokenizer_path=model_dir)
