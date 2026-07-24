from quark.torch import LLMTemplate
template = LLMTemplate(
    model_type="qwen3_6",
    q_layer_name=["*q_proj"],
    kv_layers_name=["*k_proj", "*v_proj"],
    exclude_layers_name=["*lm_head"]
)

LLMTemplate.register_template(template)
template = LLMTemplate.get("qwen3_6")

print(template.get_supported_schemes())