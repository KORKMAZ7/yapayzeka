from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# FastAPI uygulamasını oluştur
app = FastAPI()

# Model ve tokenizer'ı yükle
MODEL_NAME = "TheBloke/Llama-1B-GGUF"  # Örnek model ismi, kendi modelini kullan
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

@app.post("/generate/")
async def generate_text(prompt: str, max_length: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=max_length)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"response": response_text}

# API'yi çalıştır
# Terminalde: uvicorn app:app --host 0.0.0.0 --port 8000
