import os
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = os.path.join("models", "tinyllama_fake_app_detector")

class RiskModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def format_data_for_llm(self, df):
        # Print balance to ensure equal Real vs Fake
        print(f"📊 Training Data: {len(df[df['label']==0])} Real, {len(df[df['label']==1])} Fake")
        formatted_samples = []
        for _, row in df.iterrows():
            label_str = "FAKE" if row['label'] == 1 else "REAL"
            text = f"""<|system|>
You are a security analyst. Classify this app as REAL or FAKE.</s>
<|user|>
Name: {row['name']}
Desc: {row['desc']}
Perms: {row['perms']}
Pub: {row['publisher']}</s>
<|assistant|>
{label_str}</s>"""
            formatted_samples.append({"text": text})
        return Dataset.from_list(formatted_samples)

    def train(self, df):
        print(f"🚀 Loading {MODEL_ID}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32, 
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # STRONGER CONFIGURATION
        peft_config = LoraConfig(
            r=64,           # Increased Rank (Smarter)
            lora_alpha=128, # Increased Alpha (Stronger updates)
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
            task_type="CAUSAL_LM",
            bias="none",
            lora_dropout=0.05
        )

        dataset = self.format_data_for_llm(df)

        training_args = SFTConfig(
            output_dir=OUTPUT_DIR,
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_steps=200,      # INCREASED: 200 Steps for deeper learning
            learning_rate=3e-4, 
            fp16=False,             
            bf16=False,            
            optim="adamw_torch",
            packing=False,
            dataloader_num_workers=0, 
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            args=training_args,
        )

        self.model.config.use_cache = False 

        print("🧠 Starting Aggressive Fine-Tuning...")
        trainer.train()
        
        print("💾 Saving Adapters...")
        trainer.model.save_pretrained(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)

    def load(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def predict(self, app_data):
        if not self.model: self.load()

        prompt = f"""<|system|>
You are a security analyst. Classify this app as REAL or FAKE.</s>
<|user|>
Name: {app_data['name']}
Desc: {app_data['description']}
Perms: {app_data['permissions']}
Pub: {app_data.get('publisher', 'Unknown')}</s>
<|assistant|>
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        fake_tokens = ["FAKE", "Fake", "fake"]
        real_tokens = ["REAL", "Real", "real"]
        
        fake_sum = 0.0
        real_sum = 0.0

        for t in fake_tokens:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1: fake_sum += probs[ids[0]].item()

        for t in real_tokens:
            ids = self.tokenizer.encode(t, add_special_tokens=False)
            if len(ids) == 1: real_sum += probs[ids[0]].item()

        if (fake_sum + real_sum) == 0: return 0.0
        return (fake_sum / (fake_sum + real_sum)) * 100

def analyze_app_ml(app_data):
    model = RiskModel()
    score = model.predict(app_data)
    if score > 85: return score, "AI Flagged as Malicious"
    elif score > 50: return score, "AI Flagged as Suspicious"
    else: return score, "AI Verified Safe"