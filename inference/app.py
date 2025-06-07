from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch

app = FastAPI()
model_path = "outputs/fine_tuned_model"

# Загрузка модели
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

class Request(BaseModel):
    user: str
    prompt: str
    max_length: int = 100

@app.post("/generate")
async def generate(request: Request):
    input_text = f"[USER:{request.user}] {request.prompt}"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=request.max_length,
        do_sample=True,
        temperature=0.7
    )
    
    return {
        "response": tokenizer.decode(outputs[0], skip_special_tokens=True)
    }