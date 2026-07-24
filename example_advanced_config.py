from quark.torch.quantization.config.config import QTensorConfig, Int8PerTensorSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PlaceholderObserver, PerTensorMinMaxObserver, PerGroupMinMaxObserver

BFLOAT16_SPEC = QTensorConfig(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)

FP8_PER_TENSOR_SPEC = QTensorConfig(dtype=Dtype.fp8_e4m3,
                                    qscheme=QSchemeType.per_tensor,
                                    observer_cls=PerTensorMinMaxObserver,
                                    is_dynamic=False)

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(observer_method="min_max",
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False).to_quantization_spec()

UINT4_PER_GROUP_ASYM_SPEC = QTensorConfig(dtype=Dtype.uint4,
                                          observer_cls=PerGroupMinMaxObserver,
                                          symmetric=False,
                                          scale_type=ScaleType.float,
                                          round_method=RoundType.half_even,
                                          qscheme=QSchemeType.per_group,
                                          ch_axis=1,
                                          is_dynamic=False,
                                          group_size=128)

from quark.torch.quantization.config.config import QLayerConfig

W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                             weight=FP8_PER_TENSOR_SPEC)

W_INT8_A_INT8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                               weight=INT8_PER_TENSOR_SPEC)

W_UINT4_PER_GROUP_CONFIG = QLayerConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)

from quark.torch.quantization.config.config import AWQConfig, SmoothQuantConfig, GPTQConfig, RotationConfig, QronosConfig

ALGORITHM_CONFIG=AWQConfig(
  scaling_layers=[
    {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'},
    {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'},
    {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'},
    {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',  'help': 'linear 2'}],
  model_decoder_layers='model.layers')

ALGORITHM_CONFIG=SmoothQuantConfig(
  alpha=0.5,
  scale_clamp_min=0.001,
  scaling_layers=[
    {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'},
    {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'},
    {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'},
    {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',   'help': 'linear 2'}],
  model_decoder_layers='model.layers')

ALGORITHM_CONFIG = GPTQConfig(
    damp_percent=0.01,
    desc_act=True,
    static_groups=True,
    true_sequential=True,
    inside_layer_modules=['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'],
    model_decoder_layers='model.layers'
)

ALGORITHM_CONFIG = RotationConfig(
    model_decoder_layers="model.layers",
    scaling_layers = {
        "first_layer": [
            {"prev_modules": ["model.embed_tokens"],
             "norm_module": "model.layers.layer_id.input_layernorm",
             "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]},
            {"prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
             "norm_module": "model.layers.layer_id.post_attention_layernorm",
             "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]}],
        "middle_layers": [
            {"prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
             "norm_module": "model.layers.layer_id.input_layernorm",
             "next_modules": ["model.layers.layer_id.self_attn.q_proj", "model.layers.layer_id.self_attn.k_proj", "model.layers.layer_id.self_attn.v_proj"]},
            {"prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
             "norm_module": "model.layers.layer_id.post_attention_layernorm",
             "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]}],
        "last_layer": [
            {"prev_modules": ["model.layers.layer_id.mlp.down_proj"],
             "norm_module": "model.norm",
             "next_modules": ["lm_head"]}]
    }
)

ALGORITHM_CONFIG = QronosConfig(
    inside_layer_modules=['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'],
    model_decoder_layers='model.layers',
    block_size=128,
    desc_act=True,
    static_groups=True,
    alpha=1e-3,
    beta=1e4
)

# Example 1: W_INT8_A_INT8_PER_TENSOR
quant_config = QConfig(global_quant_config=W_INT8_A_INT8_PER_TENSOR_CONFIG)

# Example 2: W_UINT4_PER_GROUP with advanced algorithm
quant_config = QConfig(global_quant_config=W_UINT4_PER_GROUP_CONFIG, algo_config=ALGORITHM_CONFIG)
EXCLUDE_LAYERS = ["lm_head"] # For language models
quant_config = replace(quant_config, exclude=EXCLUDE_LAYERS)

# Example 3: W_FP8_A_FP8_PER_TENSOR with KV_CACHE_FP8
quant_config = QConfig(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG)
KV_CACHE_CFG = {
    "*v_proj":
    QLayerConfig(input_tensors=quant_config.global_quant_config.input_tensors,
                 weight=quant_config.global_quant_config.weight,
                 output_tensors=FP8_PER_TENSOR_SPEC),
    "*k_proj":
    QLayerConfig(input_tensors=quant_config.global_quant_config.input_tensors,
                 weight=quant_config.global_quant_config.weight,
                 output_tensors=FP8_PER_TENSOR_SPEC),
}
quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG)