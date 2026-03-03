import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_safety_percentage(metadata):
    model_path = "./tinyllama_weights"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    prompt = f"Analyze metadata: {metadata}. Safety percentage:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    outputs = model.generate(**inputs, max_new_tokens=10)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    match = re.search(r'\d+', response.replace(prompt, ""))
    if match:
        return float(match.group())
    return 0.0