from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors

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
        attn_implementation="eager",
        torch_dtype="auto",
    )
    return model.eval().to(device)


def quantize_model_pipeline(
    model: PreTrainedModel,
    calib_dataloader: DataLoader,
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedModel:

    template = LLMTemplate(
        model_type="qwen3_5",
        kv_layers_name=["*k_proj", "*v_proj"],
        q_layer_name="*q_proj",
        exclude_layers_name=["lm_head"],
    )

    LLMTemplate.register_template(template)
    template = LLMTemplate.get("qwen3_5")
    quant_config = template.get_config(scheme="fp8", kv_cache_scheme="fp8")

    quantizer = ModelQuantizer(quant_config, multi_device=True)
    quantized_model: PreTrainedModel = quantizer.quantize_model(model, calib_dataloader)

    print("[INFO] Export Quant Model.")
    export_safetensors(model=quantized_model, output_dir="./Qwen3.5-4B_fp8")
    tokenizer.save_pretrained("./Qwen3.5-4B_fp8")

    return quantized_model

@torch.no_grad()
def ppl_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: str | None,
) -> torch.Tensor:
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids.to(device)

    seqlen_for_eval = 2048
    nsamples = testenc.numel() // seqlen_for_eval
    nlls: list[torch.Tensor] = []

    for i in tqdm(range(nsamples)):
        batch = testenc[:, i * seqlen_for_eval : (i + 1) * seqlen_for_eval]
        lm_logits = model(batch)["logits"]

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:]

        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        nlls.append(loss.float() * seqlen_for_eval)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))
    return ppl

def run_quark_fp8_example() -> None:
    model_id = "Qwen/Qwen3.5-4B"
    batch_size, seq_len = 4, 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading model: {model_id}")
    model = get_model(model_id, device)
    tokenizer = get_tokenizer(model_id, max_seq_len=seq_len)
    calib_dataloader = get_dataloader(tokenizer, batch_size, device, seq_len)

    print("[INFO] Starting quantization...")
    quantized_model = quantize_model_pipeline(model, calib_dataloader, tokenizer)
    print("[INFO] Quantization complete.")
    # quantized_model = get_model("Qwen3.5-4B_fp8", device)
    print("[INFO] Simple test PPL with wikitext-2.")
    original_ppl = ppl_eval(model, tokenizer, device)
    quantized_ppl = ppl_eval(quantized_model, tokenizer, device)
    print(f"[INFO] Perplexity of the original model: {original_ppl.item():.8f}")
    print(f"[INFO] Perplexity of the quantised model: {quantized_ppl.item():.8f}")

if __name__ == "__main__":
    with torch.no_grad():
        run_quark_fp8_example()