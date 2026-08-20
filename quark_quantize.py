import math
import torch
from datasets import load_dataset
from tqdm import tqdm
from typing import Any
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from quark.torch.quantization.config.config import load_quant_algo_config_from_file
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors, export_gguf

from quark.torch.quantization.config.config import QTensorConfig, Int8PerTensorSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PlaceholderObserver, PerTensorMinMaxObserver, PerGroupMinMaxObserver, PerTensorMSEObserver, PerChannelMinMaxObserver
from quark.torch.quantization.config.config import QLayerConfig,QConfig
from dataclasses import replace

def get_pileval(
    tokenizer: PreTrainedTokenizer,
    nsamples: int,
    seqlen: int,
    device: str | None,
    seed: int = 0,
) -> torch.Tensor:
    dataset: Any = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)
    samples, n_run = [], 0

    for data in dataset:
        line_encoded = tokenizer.encode(data["text"].strip())
        if 0 < len(line_encoded) <= seqlen:
            samples.append(torch.tensor([line_encoded], device=device))
            n_run += 1
        if n_run == nsamples:
            break

    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen
    train_dataset = [cat_samples[:, i * seqlen : (i + 1) * seqlen] for i in range(n_split)]

    return torch.cat(train_dataset, dim=0)


def get_tokenizer(model_id: str, max_seq_len: int = 512) -> PreTrainedTokenizer:
    print(f"Initializing tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    assert tokenizer.pad_token is not None, "Pad token cannot be set!"
    return tokenizer


def get_dataloader(
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    device: str | None,
    seq_len: int = 512,
) -> DataLoader:
    samples: torch.Tensor = get_pileval(tokenizer, nsamples=128, seqlen=seq_len, device=device, seed=42)
    return DataLoader(samples, batch_size=batch_size, shuffle=False, drop_last=True)

def get_model(model_id: str, device: str | None) -> PreTrainedModel:
    model: PreTrainedModel = AutoModelForImageTextToText.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        dtype="auto"
    )
    return model.eval().to(device)


def quantize_model_pipeline(
    model: PreTrainedModel,
    calib_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedModel:
    # custom_autosmoothquant_config = load_quant_algo_config_from_file("qwen3_6_autosmoothquant_config.json")
    # template = LLMTemplate(
    #     model_type="qwen3_6",
    #     kv_layers_name=["*k_proj", "*v_proj"],
    #     q_layer_name="*q_proj",
    #     exclude_layers_name=["lm_head"],
    #     # algorithm_configs={"autosmoothquant": custom_autosmoothquant_config}
    # )
    STATIC_FP8_PER_TENSOR_SPEC = QTensorConfig(dtype=Dtype.fp8_e4m3,
                                    qscheme=QSchemeType.per_tensor,
                                    observer_cls=PerTensorMinMaxObserver,
                                    symmetric=True,
                                    round_method=RoundType.half_even,
                                    scale_type=ScaleType.float,
                                    is_dynamic=False)

    DYNAMIC_FP8_PER_CHANNEL_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_channel,
        ch_axis=0,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        round_method=RoundType.half_even,
        scale_type=ScaleType.float,
        is_dynamic=True,
    )

    W_FP8_A_FP8_PER_TENSOR_CONFIG = QLayerConfig(input_tensors=DYNAMIC_FP8_PER_CHANNEL_SPEC,
                                                weight=STATIC_FP8_PER_TENSOR_SPEC)

    # ALGORITHM_CONFIG=SmoothQuantConfig(
    # alpha=0.5,
    # scale_clamp_min=0.001,
    # scaling_layers=[
    #     {'prev_op': 'input_layernorm', 'layers': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'], 'inp': 'self_attn.q_proj', 'module2inspect': 'self_attn'},
    #     {'prev_op': 'self_attn.v_proj', 'layers': ['self_attn.o_proj'], 'inp': 'self_attn.o_proj'},
    #     {'prev_op': 'post_attention_layernorm', 'layers': ['mlp.gate_proj', 'mlp.up_proj'], 'inp': 'mlp.gate_proj', 'module2inspect': 'mlp', 'help': 'linear 1'},
    #     {'prev_op': 'mlp.up_proj', 'layers': ['mlp.down_proj'], 'inp': 'mlp.down_proj',   'help': 'linear 2'}],
    # model_decoder_layers='model.layers')

    quant_config = QConfig(global_quant_config=W_FP8_A_FP8_PER_TENSOR_CONFIG, exclude=["lm_head"])

    # KV_CACHE_CFG = {
    #     "*q_proj": QLayerConfig(
    #         input_tensors=quant_config.global_quant_config.input_tensors,
    #         weight=quant_config.global_quant_config.weight,
    #     ),
    #     "*k_proj": QLayerConfig(
    #         input_tensors=quant_config.global_quant_config.input_tensors,
    #         weight=quant_config.global_quant_config.weight,
    #     ),
    #     "*v_proj": QLayerConfig(
    #         input_tensors=quant_config.global_quant_config.input_tensors,
    #         weight=quant_config.global_quant_config.weight,
    #     ),
    # }
    # quant_config = replace(quant_config, layer_quant_config=KV_CACHE_CFG)

    # LLMTemplate.register_template(template)
    # template = LLMTemplate.get("qwen3_6")
    # quant_config = template.get_config(scheme="ptpc_fp8", kv_cache_scheme="fp8")

    quantizer = ModelQuantizer(quant_config, multi_device=True)
    quantized_model: PreTrainedModel = quantizer.quantize_model(model, calib_dataloader)

    print("[INFO] Export Quant Model.")
    quantized_model_dir = "models/Qwen3.6-27B-CFP8-SWDA"
    export_safetensors(model=quantized_model, output_dir=quantized_model_dir)
    tokenizer.save_pretrained(quantized_model_dir)

    return quantized_model

@torch.no_grad()
def ppl_eval(model, tokenizer, device: str | None, max_length: int = 4096, stride: int = 512) -> float:
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
        num_valid_tokens = (target_ids != -100).sum().item()

        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += (neg_log_likelihood.item() * num_loss_tokens)
        n_tokens += num_loss_tokens
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)

    return ppl

def run_quark_fp8_example() -> None:
    model_id = "Qwen/Qwen3.6-27B"
    batch_size, seq_len = 4, 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading model: {model_id}")
    model = get_model(model_id, device)
    tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)
    calib_dataloader = get_dataloader(tokenizer, batch_size, device, seq_len)

    print("[INFO] Starting quantization...")
    quantized_model = quantize_model_pipeline(model, calib_dataloader, tokenizer)
    print("[INFO] Quantization complete.")
    # print("[INFO] Simple test PPL with wikitext-2.")
    # quantized_ppl = ppl_eval(quantized_model, tokenizer, device)
    # print(f"[INFO] Perplexity of the quantised model: {quantized_ppl:.8f}")

if __name__ == "__main__":
    with torch.no_grad():
        run_quark_fp8_example()