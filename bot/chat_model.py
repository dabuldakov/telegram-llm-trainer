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

    def generate(self, promt):
        inputs = self.tokenizer(promt, return_tensors="pt", add_special_tokens=True).to("cuda")
        data = {k: v.to(self.model.device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **data,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )[0]

        self.log_output_ids(output_ids, data)

        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids)
        return output.strip()
    
    def log_output_ids(self, output_ids, data):
        output_ids = output_ids[len(data["input_ids"][0]):]
        decoded_text = self.tokenizer.decode(output_ids)
        with open(f"{logs_dir}/output_ids.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat()} output_ids: {output_ids.tolist()}\n")
            f.write(f"{datetime.datetime.now().isoformat()} detokenized: {decoded_text}\n")