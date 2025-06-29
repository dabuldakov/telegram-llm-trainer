from collections import defaultdict
import datetime
from config import Config

max_chat_history = Config.MAX_CHAT_HISTORY

class ChatHistory:
    def __init__(self):
        self.history = defaultdict(list)
        self.history_full = defaultdict(list)
        self.max_history = max_chat_history
    
    def add_message(self, chat_id, role, name, content):
        self.history[chat_id].append({"role": role, "name": name, "content": content})
        self.history_full[chat_id].append({"role": role, "name": name, "content": content, 
                                           "date": datetime.datetime.now().isoformat()})
        self.trim_history(chat_id)

    def trim_history(self, chat_id):     
        if len(self.history[chat_id]) > self.max_history:
            self.history[chat_id] = self.history[chat_id][-self.max_history:]      
    
    def get_answer_message_by_id(self, chat_id, message_id):
        for msg in self.history_answers[chat_id]:
            if msg.get("message_id") == message_id:
                return msg
        return None
    
    def get_formatted_history(self, chat_id):
        prompt = ""
        for msg in self.history[chat_id]:
            role = msg['role']
            name = msg['name']
            content = msg['content']
            prompt += f"<|{role}|>{name}|>{content}</|{role}|>\n"
        return prompt
    
    def get_formatted_history_last_day(self, chat_id):
        prompt = ""
        now = datetime.datetime.now()
        one_day_ago = now - datetime.timedelta(days=1)
        for msg in self.history_full[chat_id]:
            # Парсим дату сообщения
            msg_date = msg.get("date")
            if msg_date:
                try:
                    msg_dt = datetime.datetime.fromisoformat(msg_date)
                except Exception:
                    continue
                if msg_dt >= one_day_ago:
                    role = msg['role']
                    name = msg['name']
                    content = msg['content']
                    prompt += f"<|{role}|>{name}|>{content}</|{role}|>\n"
        return prompt