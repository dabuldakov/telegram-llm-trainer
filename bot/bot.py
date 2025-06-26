import telebot
import datetime
import random
from bot.chat_model import ChatModel
from config import Config
from bot.chat_history import ChatHistory

# Инициализация
bot = telebot.TeleBot(Config.TELEGRAM_BOT_TOKEN)
chat_model = ChatModel()
history = ChatHistory()
bot_name = "bot"
role_assistant = "assistant"
role_user = "user"
imitator_name = "Timur Mukhtarov"
logs_dir = Config.TRAINING_LOGS_PATH
bot_special_name = 'ochen_hueviy_bot'

# Загружаем фразы один раз при старте
with open("bot/warhammer_frazes.txt", encoding="utf-8") as f:
    warhammer_phrases = [line.strip() for line in f if line.strip()]

@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, "Здарово заебал!")

@bot.message_handler(commands=['help'])
def help_command(message):
    bot.send_message(message.chat.id, "Хули ноешь!")

@bot.message_handler(commands=['echo'])
def echo_command(message):
    bot.send_message(message.chat.id, "Эхо ебать!")    

@bot.message_handler(commands=['emperor'])
def emperor_command(message):
    try:
        phrase = random.choice(warhammer_phrases)
        chat_id = message.chat.id
        history.add_message(chat_id, role_user, bot_name, phrase)
        bot.send_message(chat_id, phrase)
    except Exception as e:
        bot.send_message(message, f"Ошибка при получении фразы: {str(e)}")  

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        chat_id = message.chat.id
        user_message = message.text
        #loggin_promt(message)

        # Проверка на название чата
        allowed_titles = ["Группа хуюпа", "Хуйня"]
        if not hasattr(message, "chat") or getattr(message.chat, "title", None) not in allowed_titles:
            return
        
        u_m = user_message.replace("@ochen_hueviy_bot", "").strip()
        history.add_message(chat_id, role_user, get_fio(message), u_m)
        
        # Если это reply-сообщение, то обрабатываем его отдельно
        if handle_with_reply(message):
            return

        # Если не упомянули бота то просто слушаем
        if "@ochen_hueviy_bot" not in user_message:
            return 
       
        # Если упомянули бота, то обрабатываем ответ
        handle_reply(message)
        
    except Exception as e:
        bot.reply_to(message, f"Ой произошла ошибка: {str(e)}")

def handle_reply(message):
    chat_id = message.chat.id

    discusion = history.get_formatted_history(message.chat.id)
    prompt = f"{discusion}<|assistant|>{imitator_name}|>"

    # Генерируем ответ
    loggin_promt(prompt)
    output = chat_model.generate(prompt)
    history.add_message(chat_id, role_assistant, imitator_name, output)
    bot.reply_to(message, output)

def handle_with_reply(message):
    chat_id = message.chat.id
    loggin_promt(message)
    if hasattr(message, "reply_to_message") and message.reply_to_message:
        loggin_promt(message.reply_to_message)
        if hasattr(message.reply_to_message, "json") and message.reply_to_message.json:
            loggin_promt(message.reply_to_message.json)
            if hasattr(message.reply_to_message.json, "from"):
                from_ = getattr(message.reply_to_message.json, "from")
                loggin_promt(from_)
                if from_.is_bot and from_.username == bot_special_name:    
                    # Получаем сообщение из истории по reply_to_message id
                    reply_to_msg = message.reply_to_message.json.text
                    loggin_promt(reply_to_msg)
                    if reply_to_msg:
                        # Формируем контекст из найденного сообщения
                        context = get_formatted_to_answer_context(role_assistant, imitator_name, reply_to_msg)
                        user_message_answer = message.text.replace(f"@{bot_special_name}", "").strip()
                        prompt = f"{context}\n<|user|>{get_fio(message)}|>{user_message_answer}</|user|>\n<|assistant|>{imitator_name}|>"

                        loggin_promt(prompt)
                        output = chat_model.generate(prompt)
                        history.add_message(chat_id, role_assistant, imitator_name, output)
                        bot.reply_to(message, output)
                        return True
    return False     

def get_fio(message):
    return f"{message.from_user.first_name} {message.from_user.last_name}"

def get_formatted_to_answer_context(role, name, content):
    return f"<|{role}|>{name}|>{content}</|{role}|>\n"

def loggin_promt(prompt):
    with open(f"{logs_dir}/prompts.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()}\n{prompt}\n{'-'*40}\n")       

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()