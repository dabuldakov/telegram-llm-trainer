from collections import defaultdict
from config import Config

class ChatHistory:
    def __init__(self):
        self.history = defaultdict(list)
        self.max_history = getattr(Config, "MAX_CHAT_HISTORY", 10)
    
    def add_message(self, chat_id, role, content):
        self.history[chat_id].append({"role": role, "content": content})
        self.trim_history(chat_id)

    def trim_history(self, chat_id):     
        if len(self.history[chat_id]) > self.max_history + 1:
            self.history[chat_id] = self.history[chat_id][-self.max_history:]   
    
    def get_formatted_history(self, chat_id):
        prompt = ""
        for msg in self.history[chat_id]:
            prompt += f"[USER:{msg['role']}] {msg['content']}\n"
        return prompt