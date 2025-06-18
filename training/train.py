import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_from_disk
from huggingface_hub import login
import torch
import wandb
from config import Config
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Конфигурация
model_name = "mistralai/Mistral-7B-v0.1"
dataset_path = Config.DATA_SET_PATH
output_dir = Config.MODEL_PATH
logs_dir = Config.TRAINING_LOGS_PATH
token = Config.HUGGINGFACE_TOKEN
wandb_token = Config.WANDB_TOKEN
context_length = 512

torch._dynamo.config.suppress_errors = True

# Authorization
login(token, add_to_git_credential=False)
wandb.login(key=wandb_token)

# Загрузка данных
dataset = load_from_disk(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="right",
    model_max_length=context_length,
    use_fast=True
    )
tokenizer.pad_token = tokenizer.eos_token

# Конфигурация 4-bit квантования
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 8-bit для сохранения качества
    llm_int8_threshold=6.0,
    torch_dtype=torch.bfloat16
)

# Токенизация данных
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        max_length=context_length,
        stride=256,  # Перекрытие для сохранения контекста
        padding="max_length",
        return_overflowing_tokens=True
        )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],  # Удаляем исходный текст
    num_proc=8  # Максимальный параллелизм
)

def logging_length(tokenized_dataset):
    lengths = [len(x["input_ids"]) for x in tokenized_dataset]
    with open(f"{logs_dir}/tokens.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} Средняя длина: {sum(lengths)/len(lengths)}")  

def loggin_tokens(tokenized_dataset):
    sample = tokenized_dataset[0]
    with open(f"{logs_dir}/tokens.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} Токенизированные данные: {sample}\n")  
        f.write(f"{datetime.datetime.now().isoformat()} Декодированный текст: {tokenizer.decode(sample['input_ids'])}\n")  

loggin_tokens(tokenized_dataset)
logging_length(tokenized_dataset)

# DataCollator для языкового моделирования
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Для causal LM
    pad_to_multiple_of=64
)

# Расширенная LoRA конфигурация
peft_config = LoraConfig(
    r=16,  # Увеличенный rank для качества
    lora_alpha=32,
    target_modules=[                # Ключевые слои для адаптации
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",               # Важно для Mistral
        "up_proj",                 # Дополнительные слои
        "down_proj"                # для лучшего качества
    ],
    lora_dropout=0.02,             # Регуляризация
    bias="lora_only",              # Только LoRA bias
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"],  # Сохраняем ключевые слои
    inference_mode=False
)

# Подготовка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_flash_attention_2=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

model = get_peft_model(model, peft_config)

# Принудительная проверка градиентов
for name, param in model.named_parameters():
    if "lora" in name:  # Только LoRA параметры должны иметь градиенты
        param.requires_grad = True
    else:
        param.requires_grad = False

# Параметры обучения
training_args = TrainingArguments(

    # Директории
    output_dir=output_dir,
    logging_dir=logs_dir,

    # Распределённое обучение
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=8,  

    # Оптимизация памяти
    bf16=True,                       # A100 с bfloat16
    gradient_checkpointing=True,     # Обязательно для 7B!
    torch_compile=True,              # Ускорение на Ampere (A100)
    group_by_length=True,            # Улучшает эффективность паддинга

    # Параметры обучения
    learning_rate=1e-5,
    num_train_epochs=3,
    max_grad_norm=0.5,
    warmup_ratio=0.1,
    weight_decay=0.005,  # Меньше регуляризация
    lr_scheduler_type="cosine_with_restarts",
    optim="adamw_torch_fused",

    # Логирование
    logging_steps=50,
    report_to="wandb",

    # Сохранение
    save_steps=2000,          
    save_total_limit=2,

    remove_unused_columns=False,
    dataloader_num_workers=8
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