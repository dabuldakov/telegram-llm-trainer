from datasets import Dataset
import json
import os

data_save_path = "chat_history_prepared.json"
data_save_path_individual = "chat_history_prepared_Timur Mukhtarov.json"
text_data_for_llm_save_path = "text_data_for_llm.txt"
individual_name = "Timur Mukhtarov"
output_dir = "data"

def process_chat_data(input_file):
    with open(input_file) as f:
        chat_data = json.load(f)

    json_with_context = build_context_reply_dataset(chat_data)

    save_json_with_context(json_with_context)

    json_with_context_individual = filter_by_author(
                                                    f"{output_dir}/{data_save_path}", 
                                                    f"{output_dir}/{data_save_path_individual}", 
                                                    individual_name)    

    # Форматируем тексты для подачи в ЛЛМ
    formatted_texts = [format_example_for_llm(example) for example in json_with_context_individual]

    save_text_data_for_llm(formatted_texts)

    statistic(formatted_texts)

    # Создаем и сохраняем датасет
    dataset = Dataset.from_dict({"text": formatted_texts})
    dataset.save_to_disk(f"{output_dir}/processed_dataset")


def build_context_reply_dataset(chat_json):
    # Индексируем сообщения по id для быстрого доступа
    id_to_msg = {msg["id"]: msg for msg in chat_json["messages"] if msg.get("type") == "message"}
    dataset = []

    for msg in chat_json["messages"]:
        if (
            msg.get("type") == "message"
            and "reply_to_message_id" in msg
            and msg.get("text")
            and msg["reply_to_message_id"] in id_to_msg
            and not has_link(msg.get("text"))  # Исключаем target с ссылкой
        ):
            # Собираем цепочку контекста до текущего сообщения
            context_msgs = []
            current_id = msg["reply_to_message_id"]
            # Идем по цепочке reply_to_message_id вверх
            while current_id in id_to_msg:
                prev_msg = id_to_msg[current_id]
                prev_text = prev_msg.get("text")
                if has_link(prev_text):  # Исключаем из контекста сообщения с ссылкой
                    break
                author = prev_msg.get("from", "Unknown")
                if isinstance(prev_text, list):
                    prev_text = "".join([t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in prev_text])
                context_msgs.append(f"{author}: {prev_text}")
                if "reply_to_message_id" in prev_msg:
                    current_id = prev_msg["reply_to_message_id"]
                else:
                    break
            # Контекст собирается от последнего к первому, поэтому разворачиваем
            context_msgs = context_msgs[::-1]
            context = "[SEPARATOR]".join(context_msgs)
            # Целевой ответ
            target = msg.get("text")
            if isinstance(target, list):
                target = "".join([t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in target])
            author = msg.get("from", "Unknown")
            dataset.append({
                "context": context,
                "target": target,
                "author": author
            })
    return dataset

def has_link(text):
    if not text:
        return False
    if isinstance(text, list):
        text = " ".join([t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in text])
    return "http://" in text or "https://" in text

def statistic(texts):
    bins = {
        "0-100": 0,
        "100-250": 0,
        "250-500": 0,
        "500-1000": 0,
        "1000-1500": 0,
        "1500-2000": 0,
        "2000-2500": 0,
        "2500-3000": 0,
        "3000-3500": 0,
        "3500-4000": 0,
        "4000-4500": 0,
        "4500-5000": 0,
        "5000+": 0
    }
    max_len = 0
    max_text = ""
    for text in texts:
        l = len(text)
        if l > max_len:
            max_len = l
            max_text = text
        if l <= 100:
            bins["0-100"] += 1
        elif l <= 250:
            bins["100-250"] += 1
        elif l <= 500:
            bins["250-500"] += 1
        elif l <= 1000:
            bins["500-1000"] += 1
        elif l <= 1500:
            bins["1000-1500"] += 1
        elif l <= 2000:
            bins["1500-2000"] += 1
        elif l <= 2500:
            bins["2000-2500"] += 1
        elif l <= 3000:
            bins["2500-3000"] += 1
        elif l <= 3500:
            bins["3000-3500"] += 1
        elif l <= 4000:
            bins["3500-4000"] += 1
        elif l <= 4500:
            bins["4000-4500"] += 1
        elif l <= 5000:
            bins["4500-5000"] += 1
        else:
            bins["5000+"] += 1

    print("Статистика по длине сообщений:")
    for k, v in bins.items():
        print(f"{k}: {v}")
    print(f"\nСамая длинная строка ({max_len} символов)\n")  

def format_example_for_llm(example):
    context = example["context"]
    target = example["target"]
    author = example["author"]
    lines = context.split("[SEPARATOR]")
    formatted = []
    for line in lines:
        if ":" in line:
            name, rest = line.split(":", 1)
            rest2 = rest.replace("\n", " ").strip()
            formatted.append(f"<|user|>{name}|>{rest2}</|user|>")
        else:
            formatted.append(f"<|user|>{line}</|user|>")
    target2 = target.replace("\n", " ").strip()        
    formatted.append(f"<|assistant|>{author}|>{target2}</|assistant|>")
    return "".join(formatted)     

def save_text_data_for_llm(formatted_texts):
    texts_file_path = os.path.join(output_dir, text_data_for_llm_save_path)
    with open(texts_file_path, "w", encoding="utf-8") as f:
        for text in formatted_texts:
            f.write(text + "\n")  

def save_json_with_context(json_with_context):
    texts_json_path = os.path.join(output_dir, data_save_path)
    with open(texts_json_path, "w", encoding="utf-8") as f:
        json.dump(json_with_context, f, ensure_ascii=False, indent=2)            

def filter_by_author(input_json_path, output_json_path, author_name):
    """
    Фильтрует записи по имени автора и сохраняет результат в новый файл.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [item for item in data if item.get("author") == author_name]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    return filtered

# Пример использования
process_chat_data("data/chat_history.json")