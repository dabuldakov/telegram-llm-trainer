from huggingface_hub import login
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from config import Config

MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."
token = Config.HUGGINGFACE_TOKEN

# Initialize model components globally
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Authorization
login(token=token, add_to_git_credential=False)

# Load model and tokenizer once at startup
config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

class ChatModelSaigaMistral:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_template=DEFAULT_RESPONSE_TEMPLATE
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.system_prompt = system_prompt
        self.reset_conversation()

    def reset_conversation(self):
        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()

def generate_saiga(prompt):
    """Simplified generation function that only requires the prompt text"""
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()

# Example usage
if __name__ == "__main__":
    print("Generation config:", generation_config)
    
    inputs = [
        "Почему трава зеленая?", 
        "Сочини длинный рассказ, обязательно упоминая следующие объекты. Дано: Таня, мяч"
    ]
    
    for inp in inputs:
        conversation = ChatModelSaigaMistral()
        conversation.add_user_message(inp)
        prompt = conversation.get_prompt()

        output = generate_saiga(prompt)  # Simplified call
        print("Input:", inp)
        print("Output:", output)
        print("\n" + "="*50 + "\n")