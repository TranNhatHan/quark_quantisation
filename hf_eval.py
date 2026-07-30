import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForImageTextToText

@torch.no_grad()
def ppl_eval(
    model,
    tokenizer,
    device: str | None,
    seqlen_for_eval: int = 2048,
    eval_batch_size: int = 4,
) -> torch.Tensor:
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]),return_tensors="pt").input_ids.to(device)

    nsamples = testenc.numel() // seqlen_for_eval
    total_nll = 0.0
    total_tokens = 0
    for start_idx in tqdm(range(0, nsamples, eval_batch_size)):
        end_idx = min(start_idx + eval_batch_size, nsamples)
        batch = torch.cat([testenc[:,i * seqlen_for_eval : (i + 1) * seqlen_for_eval] for i in range(start_idx, end_idx)],dim=0)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        num_tokens = shift_labels.numel()
        total_nll += loss.float() * num_tokens
        total_tokens += num_tokens
    ppl = torch.exp(total_nll / total_tokens)
    return ppl

@torch.no_grad()
def generate_summaries(model, tokenizer, device, num_samples=100, batch_size=8):
    print(f"[INFO] Generating summaries for {num_samples} articles...")

    if num_samples==-1:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test")
    else:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{num_samples}]")

    predictions = []
    references = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
        batch = dataset[i:i + batch_size]

        conversations = []

        for article in batch["article"]:
            conversations.append([
                {
                    "role": "user",
                    "content": f"Summarize the following news article:\n\n{article}"
                }
            ])

        prompts = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)

        input_lengths = inputs.attention_mask.sum(dim=1)

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        batch_preds = []

        for j in range(outputs.size(0)):
            generated = outputs[j, input_lengths[j]:]
            batch_preds.append(tokenizer.decode(generated, skip_special_tokens=True).strip())

        predictions.extend(batch_preds)
        references.extend(batch["highlights"])

    return predictions, references

def run_quark_fp8_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "Qwen3.6-27B-fp8"

    print("[INFO] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", attn_implementation="sdpa").eval().to(device)

    ppl = ppl_eval(model, tokenizer, device)

    print(f"[INFO] Perplexity: {ppl.item():.8f}\n")

    preds, refs = generate_summaries(model, tokenizer, device, num_samples=-1, batch_size=8)

    print("[INFO] Computing ROUGE and METEOR scores...")

    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    rouge_results = rouge_metric.compute(predictions=preds, references=refs)
    meteor_results = meteor_metric.compute(predictions=preds, references=refs)

    print(f"ROUGE-1:  {rouge_results['rouge1']:.8f}")
    print(f"ROUGE-2:  {rouge_results['rouge2']:.8f}")
    print(f"ROUGE-L:  {rouge_results['rougeL']:.8f}")
    print(f"METEOR:   {meteor_results['meteor']:.8f}")


if __name__ == "__main__":
    torch.manual_seed(67)
    run_quark_fp8_example()