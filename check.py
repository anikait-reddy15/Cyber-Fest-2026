import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR = os.path.join("models", "tinyllama_fake_app_detector")

class RiskDetector:
    def __init__(self):
        print(f"⏳ Loading Model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32, 
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        self.model = PeftModel.from_pretrained(self.base_model, ADAPTER_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        print("✅ Model Loaded!\n")

    def test_app(self, name, desc, pub, perm):
        prompt = f"""<|system|>
You are a security analyst. Classify this app as REAL or FAKE.</s>
<|user|>
Name: {name}
Desc: {desc}
Perms: {perm}
Pub: {pub}</s>
<|assistant|>
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 1. TEXT GENERATION TEST (What does the model actually say?)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=5, pad_token_id=self.tokenizer.eos_token_id)
        
        raw_output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # Extract just the new part (after assistant)
        assistant_reply = raw_output.split("<|assistant|>")[-1].strip()

        # 2. PROBABILITY MATH
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        fake_id = self.tokenizer.encode("FAKE", add_special_tokens=False)[0]
        real_id = self.tokenizer.encode("REAL", add_special_tokens=False)[0]
        
        fake_score = probs[fake_id].item()
        real_score = probs[real_id].item()
        
        total = fake_score + real_score
        risk_percent = (fake_score / total * 100) if total > 0 else 0

        return assistant_reply, risk_percent

test_cases = [
    {"name": "WhatsApp Gold", "desc": "Exclusive unlocked features.", "pub": "Unknown", "perm": "READ_CONTACTS"},
    {"name": "Calculator", "desc": "Simple math tool.", "pub": "Google", "perm": "none"},
    {"name": "Free Gems", "desc": "Unlimited coins generator.", "pub": "Cheats", "perm": "SEND_SMS"},
]

if __name__ == "__main__":
    detector = RiskDetector()
    print(f"{'APP NAME':<20} | {'SAYS':<10} | {'RISK %':<8}")
    print("-" * 50)
    
    for app in test_cases:
        reply, score = detector.test_app(app['name'], app['desc'], app['pub'], app['perm'])
        print(f"{app['name']:<20} | {reply:<10} | {score:.1f}%")