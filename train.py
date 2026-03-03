import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from extractor import process_folder

def prepare_data(metadata_list):
    texts = []
    for data in metadata_list:
        prompt = f"Analyze this app metadata: {data}. Safety percentage: "
        texts.append({"text": prompt})
    return texts

def train_model():
    raw_data = process_folder()
    dataset = prepare_data(raw_data)
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(3):
        for batch in dataloader:
            inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
    model.save_pretrained("./tinyllama_weights")
    tokenizer.save_pretrained("./tinyllama_weights")

if __name__ == "__main__":
    train_model()