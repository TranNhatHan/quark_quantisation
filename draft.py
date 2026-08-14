import torch
from transformers import AutoModelForImageTextToText

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.6-27B", dtype="auto", attn_implementation="sdpa").eval().to(device)
print(model.config)
print(model.config.to_dict().keys())

print("language model config:")
print(model.config.language_model)

layer = model.model.language_model.layers[0]

print(type(layer))
print(layer)