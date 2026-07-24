from quark.torch.quantization.config.config import SmoothQuantConfig, QConfig

smoothquant_config = SmoothQuantConfig(
    scaling_layers=[{"prev_op": "layer_norm", "layers": ["lin1"], "inp": "lin1"}],
    model_decoder_layers="layers",
    alpha=0.5,
    scale_clamp_min=1e-12,
)

# There may be several algorithms, hence the list.
quant_config = QConfig(..., algo_config=[smoothquant_config])