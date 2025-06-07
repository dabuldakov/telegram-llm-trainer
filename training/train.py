from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_from_disk
import torch

# Конфигурация
model_name = "mistralai/Mistral-7B-v0.1"
dataset_path = "data/processed_dataset"
output_dir = "outputs/fine_tuned_model"

# Загрузка данных
dataset = load_from_disk(dataset_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Подготовка модели
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Параметры обучения
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Запуск обучения
trainer.train()
trainer.save_model(output_dir)