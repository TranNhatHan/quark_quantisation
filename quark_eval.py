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

model_path = "Qwen/Qwen3.6-27B"

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    use_safetensors=True,
    dtype="auto",
    device_map="auto"
)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)

testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",split="test")
testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

ppl = ppl_eval(model, testenc, device, "hf_format")

model_args = {
    "pretrained": model_path,
    "dtype": "auto",
    "trust_remote_code": True,
    "device_map": "auto"
}

generations = rouge_meteor_generations(
                        dataset="xsum",
                        model=model,
                        tokenizer=tokenizer,
                        num_eval_data=200,
                        import_file_format="hf_format",
                        import_model_dir=model_path,
                        model_args=model_args,
                        batch_size=8,
                        device=device,
                        max_new_toks = 128,
                        seq_len=512,
                    )

rouge_scores = rouge_eval("xsum", generations)
meteor_scores = meteor_eval("xsum", generations)

print(f"\n[INFO] XSum ROUGE: {rouge_scores}")
print(f"\n[INFO] XSum METEOR: {meteor_scores}")
print(f"\n[INFO] WikiText Perplexity: {ppl.item()}")