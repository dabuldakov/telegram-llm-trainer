import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from huggingface_hub import login
import torch
import wandb
from config import Config

# Конфигурация
model_name = "mistralai/Mistral-7B-v0.1"
dataset_path = Config.DATA_SET_PATH
output_dir = Config.MODEL_PATH
logs_dir = Config.TRAINING_LOGS_PATH
token = Config.HUGGINGFACE_TOKEN
wandb_token = Config.WANDB_TOKEN

# Authorization
login(token, add_to_git_credential=False)
wandb.login(key=wandb_token)

# Загрузка данных
dataset = load_from_disk(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=512)

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Удаляем исходный текст
)

def logging_length(tokenized_dataset):
    lengths = [len(x["input_ids"]) for x in tokenized_dataset]
    with open(f"{logs_dir}/tokens.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} Средняя длина: {sum(lengths)/len(lengths)}")  

def logging_tokens(tokenized_dataset):
    sample = tokenized_dataset[0]
    with open(f"{logs_dir}/tokens.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} Токенизированные данные: {sample}\n")  
        f.write(f"{datetime.datetime.now().isoformat()} Декодированный текст: {tokenizer.decode(sample['input_ids'])}\n")  

logging_tokens(tokenized_dataset)
logging_length(tokenized_dataset)


# DataCollator для языкового моделирования
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Для causal LM
)

# Подготовка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Параметры обучения
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    bf16=True,
    save_steps=2000,          
    save_total_limit=2,
    optim="adamw_torch",
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to="wandb",
    max_grad_norm=0.5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator  # Критически важно для CausalLM!
)

# Запуск обучения
trainer.train()
trainer.save_model(output_dir)