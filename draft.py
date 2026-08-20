from transformers import AutoTokenizer, AutoModelForImageTextToText

model_path = "models/Qwen3.6-27B-CFP8-SWDA" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", dtype="auto")

from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("Once upon a time,")
print(output)