from transformers import AutoModelForImageTextToText, AutoTokenizer
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype="auto")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

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
template = LLMTemplate.get("qwen3_5")
quant_config = template.get_config(scheme="fp8", kv_cache_scheme="fp8")

from quark.torch import ModelQuantizer
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

from quark.torch import export_safetensors
export_safetensors(
    model=quant_model,
    output_dir="./Qwen3.5-4B-fp8"
)
tokenizer.save_pretrained("./Qwen3.5-4B-fp8")
