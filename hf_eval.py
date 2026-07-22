import evaluate
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("Qwen3.5-4B_fp8")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen3.5-4B_fp8",
    torch_dtype="auto",
    device_map="auto"
)

device = next(model.parameters()).device

dataset = load_dataset("EdinburghNLP/xsum", split="test").select(range(100))

predictions = []
references = []

for sample in dataset:
    prompt = (
        "Summarize the following news article.\n\n"
        f"{sample['document']}\n\nSummary:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    prediction = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    predictions.append(prediction)
    references.append(sample["summary"])

rouge = evaluate.load("rouge")

results = rouge.compute(
    predictions=predictions,
    references=references,
)

print(results)