import torch
import evaluate
import math
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForImageTextToText

@torch.no_grad()
def ppl_eval(
    model,
    tokenizer,
    device: str | None,
    max_length: int = 4096,
    stride: int = 512,
) -> float:

    test = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )

    encodings = tokenizer(
        "\n\n".join(test["text"]),
        return_tensors="pt",
    )

    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)

        trg_len = end_loc - prev_end_loc

        input_ids = encodings.input_ids[
            :, begin_loc:end_loc
        ].to(device)

        target_ids = input_ids.clone()

        target_ids[:, :-trg_len] = -100

        outputs = model(
            input_ids,
            labels=target_ids,
        )

        neg_log_likelihood = outputs.loss

        num_valid_tokens = (
            target_ids != -100
        ).sum().item()

        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += (
            neg_log_likelihood.item()
            * num_loss_tokens
        )

        n_tokens += num_loss_tokens

        prev_end_loc = end_loc

        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)

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
            max_new_tokens=512,
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
    model_path = "Qwen/Qwen3.6-27B"

    print("[INFO] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype="auto", attn_implementation="sdpa").eval().to(device)

    ppl = ppl_eval(model, tokenizer, device)

    print(f"[INFO] Perplexity: {ppl:.8f}\n")

    preds, refs = generate_summaries(model, tokenizer, device, num_samples=100, batch_size=8)

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