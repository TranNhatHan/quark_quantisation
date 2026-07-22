import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForImageTextToText

@torch.no_grad()
def ppl_eval(model, tokenizer, device, seqlen_for_eval=2048):
    print("[INFO] Evaluating Perplexity on WikiText-2...")
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt").input_ids.to(device)

    nsamples = testenc.numel() // seqlen_for_eval
    nlls = []

    for i in tqdm(range(nsamples), desc="Evaluating PPL"):
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

@torch.no_grad()
def generate_summaries(model, tokenizer, device, num_samples=100, batch_size=8):
    print(f"[INFO] Generating summaries for {num_samples} CNN/DailyMail articles...")
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{num_samples}]")

    predictions = []
    references = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Summaries"):
        batch = dataset[i : i + batch_size]
        
        prompts = [f"Article: {article}\n\nSummary:" for article in batch["article"]]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024,
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_tokens = outputs[:, input_length:]

        batch_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        predictions.extend(batch_preds)
        references.extend(batch["highlights"])

    return predictions, references

def run_quark_fp8_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./Qwen3.5-4B_fp8"

    print("[INFO] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        padding_side="left" 
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        attn_implementation="sdpa",
    ).eval()

    ppl = ppl_eval(model, tokenizer, device)
    print(f"[INFO] Perplexity: {ppl.item():.4f}\n")

    preds, refs = generate_summaries(model, tokenizer, device, num_samples=100, batch_size=8)

    print("[INFO] Computing ROUGE and METEOR metrics...")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    rouge_scores = rouge.compute(predictions=preds, references=refs)
    meteor_scores = meteor.compute(predictions=preds, references=refs)

    print("[INFO] ROUGE Results:", rouge_scores)
    print("[INFO] METEOR Results:", meteor_scores)


if __name__ == "__main__":
    run_quark_fp8_example()