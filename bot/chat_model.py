from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
import datetime
from config import Config

model_path = Config.MODEL_PATH_FINISHED_TRAIN
logs_dir = Config.TRAINING_LOGS_PATH

class ChatModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        data = {k: v.to(self.model.device) for k, v in inputs.items()}

        assistant_token_id = self.tokenizer.encode("</|assistant|>")[-1]
        output_ids = self.model.generate(
            **data,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            eos_token_id=assistant_token_id,
            top_p=0.9,        # 0.8-0.95 (nucleus sampling)
            top_k=50,         # Ограничивает выбор топ-K токенов
            repetition_penalty=1.2,  # Штраф за повторения (1.0-2.0)
        )[0]

        self.log_output_ids(output_ids, data)

        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids)
        return output.replace("</|assistant|>", "").strip()
    
    def generate_summury(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        data = {k: v.to(self.model.device) for k, v in inputs.items()}

        assistant_token_id = self.tokenizer.encode("</|assistant|>")[-1]
        output_ids = self.model.generate(
            **data,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            eos_token_id=assistant_token_id,
            top_p=0.9,        # 0.8-0.95 (nucleus sampling)
            top_k=50,         # Ограничивает выбор топ-K токенов
            repetition_penalty=1.2,  # Штраф за повторения (1.0-2.0)
        )[0]

        self.log_output_ids(output_ids, data)

        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids)
        return output.replace("</|assistant|>", "").strip()
    
    def log_output_ids(self, output_ids, data):
        output_ids = output_ids[len(data["input_ids"][0]):]
        decoded_text = self.tokenizer.decode(output_ids)
        with open(f"{logs_dir}/output_ids.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} output_ids: {output_ids.tolist()}\n")
            f.write(f"{datetime.datetime.now().isoformat()} detokenized: {decoded_text}\n")