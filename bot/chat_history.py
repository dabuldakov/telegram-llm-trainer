from collections import defaultdict
from config import Config

class ChatHistory:
    def __init__(self):
        self.history = defaultdict(list)
    
    def add_message(self, chat_id, role, content):
        self.history[chat_id].append({"role": role, "content": content})
        # Ограничиваем историю
        if len(self.history[chat_id]) > Config.MAX_HISTORY:
            self.history[chat_id] = self.history[chat_id][-Config.MAX_HISTORY:]
    
    def get_formatted_history(self, chat_id):
        prompt = ""
        for msg in self.history[chat_id]:
            prompt += f"[USER:{msg['role']}] {msg['content']}\n"
        return prompt