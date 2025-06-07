from datasets import Dataset
import json
import os

def process_chat_data(input_file, output_dir):
    with open(input_file) as f:
        chat_data = json.load(f)
    
    # Извлекаем только нужные сообщения
    messages = [
        msg for msg in chat_data.get("messages", [])
        if (msg.get("type") == "message" and 
            msg.get("text") and 
            "forwarded_from" not in msg)
    ]
    
    # Форматируем текст сообщений
    texts = [
        f"[USER:{msg['from']}] {msg['text']}" 
        for msg in messages
    ]

    # Сохраняем тексты в файл texts.txt
    texts_file_path = os.path.join(output_dir, "texts.txt")
    with open(texts_file_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Создаем и сохраняем датасет
    dataset = Dataset.from_dict({"text": texts})
    dataset.save_to_disk(f"{output_dir}/processed_dataset")

# Пример использования
process_chat_data("data/chat_history.json", "data")