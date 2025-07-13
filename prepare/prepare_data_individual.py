from datasets import Dataset
import json
from config import Config
from analize.analize import statistic

text_json_path = Config.CHAT_HISTORY_PREPARED_PATH
text_data_for_llm_save_path = Config.TEXT_DATA_FOR_LLM_SAVE_PATH
data_set_path = Config.DATA_SET_PATH
user_names_path = Config.DATA_USER_NAMES

def process_chat_data(input_file):
    with open(input_file) as f:
        chat_data = json.load(f)

    json_with_context = build_context_reply_dataset(chat_data)

    save_json_with_context(json_with_context)

    # Форматируем тексты для подачи в ЛЛМ
    formatted_texts = [format_example_for_llm(example) for example in json_with_context]

    save_text_data_for_llm(formatted_texts)

    statistic(formatted_texts)

    # Создаем и сохраняем датасет
    dataset = Dataset.from_dict({"text": formatted_texts})
    dataset.save_to_disk(data_set_path)


def build_context_reply_dataset(chat_json):
    # Индексируем сообщения по id для быстрого доступа
    id_to_msg = {msg["id"]: msg for msg in chat_json["messages"] if msg.get("type") == "message"}
    dataset = []
    dataset_names = set()

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
                dataset_names.add(author)  # Добавляем автора target-сообщения
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
            dataset_names.add(author)  # Добавляем автора target-сообщения
            dataset.append({
                "context": context,
                "target": target,
                "author": author
            })

    clean_names = [name for name in dataset_names if name is not None]
    with open(user_names_path, "w", encoding="utf-8") as f:
        for name in sorted(clean_names):
            f.write(name + "\n")
    return dataset

def has_link(text):
    if not text:
        return False
    if isinstance(text, list):
        text = " ".join([t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in text])
    return "http://" in text or "https://" in text

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
    with open(text_data_for_llm_save_path, "w", encoding="utf-8") as f:
        for text in formatted_texts:
            f.write(text + "\n")  

def save_json_with_context(json_with_context):
    with open(text_json_path, "w", encoding="utf-8") as f:
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