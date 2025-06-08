from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from request import Request
from config import Config

model_path = Config.MODEL_PATH

class ChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    def generate(self, request: Request):
        input_text = f"[USER:{request.user}] {request.prompt}"
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)