from collections import defaultdict
from config import Config

max_chat_history = Config.MAX_CHAT_HISTORY
max_chat_history_answers = Config.MAX_CHAT_HISTORY_ANSWERS

class ChatHistory:
    def __init__(self):
        self.history = defaultdict(list)
        self.history_answers = defaultdict(list)
        self.max_history = max_chat_history
        self.max_history_answers = max_chat_history_answers
    
    def add_message(self, chat_id, role, name, content):
        self.history[chat_id].append({"role": role, "name": name, "content": content})
        self.trim_history(chat_id)

    def add_message_answer(self, chat_id, role, name, content, message_id):
        self.history_answers[chat_id].append({"role": role, "name": name, "content": content, "message_id": message_id})
        self.trim_history_answers(chat_id)

    def trim_history(self, chat_id):     
        if len(self.history[chat_id]) > self.max_history:
            self.history[chat_id] = self.history[chat_id][-self.max_history:]   

    def trim_history_answers(self, chat_id):     
        if len(self.history_answers[chat_id]) > self.max_history_answers:
            self.history_answers[chat_id] = self.history_answers[chat_id][-self.max_history_answers:]           
    
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