from transformers import AutoModelForImageTextToText, AutoTokenizer
original_model_name="Qwen/Qwen3.6-27B"
model = AutoModelForImageTextToText.from_pretrained(original_model_name, torch_dtype="auto", )
model.eval()
tokenizer = AutoTokenizer.from_pretrained(original_model_name)

from torch.utils.data import DataLoader
text = "Hello, how are you?"
tokenized_outputs = tokenizer(text, return_tensors="pt")
calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

from quark.torch import LLMTemplate
template = LLMTemplate(
    model_type="qwen3_6",
    q_layer_name=["*q_proj"],
    kv_layers_name=["*k_proj", "*v_proj"],
    exclude_layers_name=["*lm_head"]
)

LLMTemplate.register_template(template)
template = LLMTemplate.get("qwen3_6")
quant_config = template.get_config(scheme="fp8", kv_cache_scheme="fp8")

from quark.torch import ModelQuantizer
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

from quark.torch import export_safetensors
export_model_name="./Qwen3.6-27B-fp8_without_cab"
export_safetensors(
    model=quant_model,
    output_dir=export_model_name
)
tokenizer.save_pretrained(export_model_name)
