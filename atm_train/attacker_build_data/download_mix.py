from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ministral/Ministral-3b-instruct")
model = AutoModelForCausalLM.from_pretrained("ministral/Ministral-3b-instruct")

# Save model and tokenizer in the desired directory
save_path = "pretrained_models/Lite"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

# Print the paths of saved files
print("Tokenizer files:", tokenizer.name_or_path)
print("Model files:", model.name_or_path)
'''from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.cuda.amp import autocast
torch.cuda.empty_cache()
# Your local path
torch.cuda.set_per_process_memory_fraction(0.8, 0)  # Set 80% of GPU memory for PyTorch
save_path = "pretrained_models/Mixtral"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load from local directory

tokenizer = AutoTokenizer.from_pretrained(save_path)
model = AutoModelForCausalLM.from_pretrained(save_path).to(device)#, device_map="auto")

# Quick test with mixed precision
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)

# Use autocast for mixed precision
with autocast():
    outputs = model.generate(**inputs, max_new_tokens=10)
    
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))'''