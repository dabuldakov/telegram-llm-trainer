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
from config import Config

# Конфигурация
model_name = "mistralai/Mistral-7B-v0.1"
dataset_path = Config.DATA_SET_PATH
output_dir = Config.MODEL_PATH
logs_dir = Config.TRAINING_LOGS_PATH
token = Config.HUGGINGFACE_TOKEN

# Для Mistral-7B на 2x GPU:
fsdp_config = {
    "fsdp_transformer_layer_cls_to_wrap": ["MistralDecoderLayer"],
    "fsdp_sharding_strategy": 1  # FULL_SHARD
}

# Authorization
login(token, add_to_git_credential=False)

# Загрузка данных
dataset = load_from_disk(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=256) # уменьшить для длины сообщений ???

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Удаляем исходный текст
)

def loggin_tokens(tokenized_dataset):
    sample = tokenized_dataset[0]
    with open(f"{logs_dir}/tokens.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} Токенизированные данные: {sample}\n")  
        f.write(f"{datetime.datetime.now().isoformat()} Декодированный текст: {tokenizer.decode(sample['input_ids'])}\n")  

loggin_tokens(tokenized_dataset)

# DataCollator для языкового моделирования
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Для causal LM
)

# Подготовка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Параметры обучения
training_args = TrainingArguments(

    # Директории
    output_dir=output_dir,
    logging_dir=logs_dir,

    # Распределённое обучение
    num_devices=2,                    # 2 GPU
    per_device_train_batch_size=10,   # 10 на каждом GPU → глобальный батч 20
    gradient_accumulation_steps=3,    # Эффективный батч = 10*3*2 = 60
    fsdp="full_shard auto_wrap",      # Оптимально для Mistral на 2+ GPU

    # Оптимизация памяти
    bf16=True,                       # A100 с bfloat16
    gradient_checkpointing=True,     # Обязательно для 7B!
    torch_compile=True,              # Ускорение на Ampere (A100)

    # Параметры обучения
    learning_rate=3e-5,
    num_train_epochs=2,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    weight_decay=0.01,

    # Логирование
    logging_steps=50,
    report_to="wandb",

    # Сохранение
    save_steps=2000,          
    save_total_limit=2
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