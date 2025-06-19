from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from config import Config

model_path = Config.MODEL_PATH

class ChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16)

    def generate(self, promt):
        inputs = self.tokenizer(promt, return_tensors="pt").to("cuda")
        data = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **data,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5
        )
        # Убедимся, что есть сгенерированные токены
        if len(output_ids[0]) > len(data["input_ids"][0]):
            # Берем только сгенерированные токены (исключая промпт)
            output_ids = output_ids[0][len(data["input_ids"][0]):]
            return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return ""  # Возвращаем пустую строку, если ничего не сгенерировано