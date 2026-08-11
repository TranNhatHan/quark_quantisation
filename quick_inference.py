from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Qwen/Qwen3.6-27B-FP8" 
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype="auto", trust_remote_code=True)

inputs = tokenizer("Once upon a time,", return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=50)

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)
