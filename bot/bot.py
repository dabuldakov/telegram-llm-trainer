import telebot
import datetime
import random
from bot.chat_model import ChatModel
from config import Config
from bot.chat_history import ChatHistory
from telebot.types import BotCommand

# Инициализация
bot = telebot.TeleBot(Config.TELEGRAM_BOT_TOKEN)
chat_model = ChatModel()
history = ChatHistory()
bot_name = "bot"
role_assistant = "assistant"
role_user = "user"
logs_dir = Config.TRAINING_LOGS_PATH
user_names_path = Config.DATA_USER_NAMES
bot_special_name = 'ochen_hueviy_bot'
summury_default_message = 'Ты — аналитик текста. Разбери этот диалог и выдели ключевые идеи. Кто уже обсуждал эту тему и когда. Расскажи что сам думаешь об этом. '
imitator_name = "Assistant"

# Загружаем список имён для имитации
with open(user_names_path, encoding="utf-8") as f:
    imitator_names = [line.strip() for line in f if line.strip()]

def set_random_imitator_name():
    global imitator_name
    imitator_name = random.choice(imitator_names) if imitator_names else "Ассистент"
    
set_random_imitator_name()    

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

@bot.message_handler(commands=['imitator'])
def echo_command(message):
    set_random_imitator_name()
    bot.send_message(message.chat.id, f"установлен ассистент: {imitator_name}")         

@bot.message_handler(commands=['emperor'])
def emperor_command(message):
    try:
        phrase = random.choice(warhammer_phrases)
        chat_id = message.chat.id
        history.add_message(chat_id, role_user, bot_name, phrase)
        bot.send_message(chat_id, phrase)
    except Exception as e:
        bot.send_message(message, f"Ooops, error when fraze get... {str(e)}")  

@bot.message_handler(commands=['summury'])
def summury_command(message):
    try:
        handle_summury(message)
    except Exception as e:
        bot.send_message(message, f"Oops, error when try summury... : {str(e)}")          

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
        handle_mention(message)
        
    except Exception as e:
        bot.reply_to(message, f"Oops, error when handle message... {str(e)}")

def handle_summury(message):
    chat_id = message.chat.id

    discusion = history.get_formatted_history_last_day(chat_id)
    prompt = f"### Контекст: \n {discusion} \n### Задание: {summury_default_message}"

    # Генерируем ответ
    loggin_promt(prompt)
    output = chat_model.generate_summury(prompt)
    bot.reply_to(message, output)            

def handle_mention(message):
    chat_id = message.chat.id
    discusion = history.get_formatted_history(chat_id)
    prompt = f"{discusion}<|assistant|>{imitator_name}|>"

    # Генерируем ответ
    loggin_promt(prompt)
    output = chat_model.generate(prompt)
    history.add_message(chat_id, role_assistant, imitator_name, output)
    bot.reply_to(message, f"{output} ({imitator_name})")

def handle_with_reply(message):
    chat_id = message.chat.id
    if hasattr(message, "reply_to_message") and message.reply_to_message:
        if hasattr(message.reply_to_message, "from_user") and  message.reply_to_message.from_user:
            from_user = message.reply_to_message.from_user
            if from_user.is_bot and from_user.username == bot_special_name:    
                # Получаем сообщение из истории по reply_to_message id
                reply_to_msg = message.reply_to_message.text
                if reply_to_msg:
                    # Формируем контекст из найденного сообщения
                    context = get_formatted_to_answer_context(role_assistant, imitator_name, reply_to_msg)
                    user_message_answer = message.text.replace(f"@{bot_special_name}", "").strip()
                    prompt = f"{context}\n<|user|>{get_fio(message)}|>{user_message_answer}</|user|>\n<|assistant|>{imitator_name}|>"

                    loggin_promt(prompt)
                    output = chat_model.generate(prompt)
                    history.add_message(chat_id, role_assistant, imitator_name, output)
                    bot.reply_to(message, f"{output} ({imitator_name})")
                    return True
    return False     

def get_fio(message):
    return f"{message.from_user.first_name} {message.from_user.last_name}"

def get_formatted_to_answer_context(role, name, content):
    return f"<|{role}|>{name}|>{content}</|{role}|>\n"

def loggin_promt(prompt):
    with open(f"{logs_dir}/prompts.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()}\n{prompt}\n{'-'*40}\n")

def set_commands(bot):
    commands = [
        BotCommand("/start", "Say hello"),
        BotCommand("/help", "Get help"),
        BotCommand("/echo", "Echo"),
        BotCommand("/emperor", "Gain strength in moments of weakness"),
        BotCommand("/summury", "Summurize all messages for last day"),
        BotCommand("/imitator", "Set random imitator name")
    ]
    bot.set_my_commands(commands)

if __name__ == "__main__":
    print("Бот запущен...")
    set_commands(bot)
    bot.polling()