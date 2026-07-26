from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, AutoModelForImageTextToText
from quark.contrib.llm_eval import (
    lm_eval_entrypoint,
    meteor_eval,
    mlperf_rouge_eval,
    ppl_eval,
    ppl_eval_for_kv_cache,
    rouge_eval,
    rouge_meteor_generations,
)
from datasets import load_dataset
import torch

device = torch.device("cuda")

model_path = "Qwen3.5-4B_fp8"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    use_safetensors=True,
    dtype="auto"
)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)

testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",split="test",)
testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

ppl = ppl_eval(model, testenc, device, "hf_format")
print(f"\n[INFO] Perplexity: {ppl.item()}")