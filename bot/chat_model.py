from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from request import Request
from config import Config

model_path = Config.MODEL_PATH
model_paths = Config.MODEL_PATHS

class ChatModel:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def get_model_and_tokenizer(self, chat_id):
        # Получаем путь к модели для данного чата, иначе дефолт
        path = model_paths.get(str(chat_id), Config.MODEL_PATH)
        if path not in self.models:
            self.tokenizers[path] = AutoTokenizer.from_pretrained(path)
            self.models[path] = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        return self.models[path], self.tokenizers[path]

    def generate(self, request: Request, chat_id=None):
        # chat_id обязателен для выбора модели
        model, tokenizer = self.get_model_and_tokenizer(chat_id)
        input_text = f"[USER:{request.user}] {request.prompt}"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            do_sample=True,
            temperature=0.7
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)