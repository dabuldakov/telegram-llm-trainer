import json
import os

def process_chat_data(input_file, output_dir):
    with open(input_file) as f:
        chat_data = json.load(f)
    texts = build_context_reply_dataset(chat_data)

    # Сохраняем тексты в формате JSON (каждый элемент — словарь с context, target, author)
    texts_json_path = os.path.join(output_dir, "chat_history_prepared.json")
    with open(texts_json_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)


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
        ):
            # Собираем цепочку контекста до текущего сообщения
            context_msgs = []
            current_id = msg["reply_to_message_id"]
            # Идем по цепочке reply_to_message_id вверх
            while current_id in id_to_msg:
                prev_msg = id_to_msg[current_id]
                author = prev_msg.get("from", "Unknown")
                text = prev_msg.get("text")
                if isinstance(text, list):
                    # Если text — список (иногда бывает), склеиваем только строки
                    text = "".join([t["text"] if isinstance(t, dict) and "text" in t else str(t) for t in text])
                context_msgs.append(f"{author}: {text}")
                # Если у предыдущего сообщения тоже есть reply_to_message_id, продолжаем цепочку
                if "reply_to_message_id" in prev_msg:
                    current_id = prev_msg["reply_to_message_id"]
                else:
                    break
            # Контекст собирается от последнего к первому, поэтому разворачиваем
            context_msgs = context_msgs[::-1]
            context = "\n".join(context_msgs)
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

def is_link_only(msg):
    text = msg.get("text")
    if isinstance(text, list) and text:
        return any(isinstance(t, dict) and t.get("type") in ("link", "text_link")
                   for t in text)
    return False    

def filter_by_author(input_json_path, output_json_path, author_name):
    """
    Фильтрует записи по имени автора и сохраняет результат в новый файл.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered = [item for item in data if item.get("author") == author_name]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

# Пример использования
process_chat_data("data/chat_history.json", "data")

# Пример использования:
filter_by_author("data/chat_history_prepared.json", 
                 "data/chat_history_prepared_Timur Mukhtarov.json", 
                 "Timur Mukhtarov")
