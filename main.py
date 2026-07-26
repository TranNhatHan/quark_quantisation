from quark.torch import LLMTemplate

template = LLMTemplate(
    model_type="qwen3_6",
    kv_layers_name=["*k_proj", "*v_proj"],
    q_layer_name="*q_proj",
    exclude_layers_name=["lm_head"],
)

LLMTemplate.register_template(template)
template = LLMTemplate.get("qwen3_6")
quant_config = template.get_config(scheme="ptpc_fp8", kv_cache_scheme="fp8")
print(quant_config)