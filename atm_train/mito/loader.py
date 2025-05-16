from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings

base_model_path = "/home/julien/ATM-RAG/atm_train/generator_sft/experiments/model_final"
adapter_path = "/home/julien/ATM-RAG/atm_train/mito/experiments/model_final"

# Suppress missing keys warnings for LoRA weights
warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint")

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load adapter on top of base model
model = PeftModel.from_pretrained(base_model, adapter_path)

for name, module in model.named_modules():
    if "lora" in name.lower():
        print(name, module)


# Test generation
prompt = "Hello, how are you? I'm doing well!"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
