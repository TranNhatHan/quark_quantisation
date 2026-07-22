from quark.torch import LLMTemplate

template = LLMTemplate(
    model_type="qwen3_5",
    q_layer_name=["*q_proj"],
    kv_layers_name=["*k_proj", "*v_proj"],
    exclude_layers_name=["*lm_head"]
)
LLMTemplate.register_template(template)

template = LLMTemplate.get("qwen3")
config = template.get_config("fp8")

print(config.global_quant_config.input_tensors)
print(config.global_quant_config.weight)