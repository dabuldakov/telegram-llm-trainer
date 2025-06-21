from datasets import Dataset
import json
import os

def process_chat_data(input_file, output_dir, only_one_user):
    with open(input_file) as f:
        chat_data = json.load(f)

    def is_link_only(msg):
        text = msg.get("text")
        if isinstance(text, list) and text:
            return any(isinstance(t, dict) and t.get("type") in ("link", "text_link")
                       for t in text)
        return False    
    
    # Извлекаем только нужные сообщения
    messages = [
        msg for msg in chat_data.get("messages", [])
        if (
            msg.get("type") == "message" 
            and msg.get("text") 
            and "forwarded_from" not in msg
            and not is_link_only(msg)
            and (not only_one_user or msg.get("from") == only_one_user)
            )
    ]
    
    # Форматируем текст сообщений
    texts = [
        f"[USER:{msg['from']}] {msg['text']}" 
        for msg in messages
    ]

    statistic(texts)

    # Сохраняем тексты в файл texts.txt
    texts_file_path = os.path.join(output_dir, "texts.txt")
    with open(texts_file_path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    
    # Создаем и сохраняем датасет
    dataset = Dataset.from_dict({"text": texts})
    dataset.save_to_disk(f"{output_dir}/processed_dataset")

def statistic(texts):
    # Статистика по длине сообщений
    bins = {
        "0-100": 0,
        "100-250": 0,
        "250-500": 0,
        "500-1000": 0,
        "1000+": 0
    }
    for text in texts:
        l = len(text)
        if l <= 100:
            bins["0-100"] += 1
        elif l <= 250:
            bins["100-250"] += 1
        elif l <= 500:
            bins["250-500"] += 1
        elif l <= 1000:
            bins["500-1000"] += 1
        else:
            bins["1000+"] += 1

    print("Статистика по длине сообщений:")
    for k, v in bins.items():
        print(f"{k}: {v}")        

# Пример использования
process_chat_data("data/chat_history.json", "data", "")