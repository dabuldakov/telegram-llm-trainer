import telebot
import datetime
import random
from bot.chat_model_saiga_mistral import ChatModelSaigaMistral
from bot.chat_model import ChatModel
from config import Config
from bot.chat_history import ChatHistory

# Инициализация
bot = telebot.TeleBot(Config.TELEGRAM_BOT_TOKEN)
chat_model = ChatModel()
chat_model_mistral = ChatModelSaigaMistral()
history = ChatHistory()
bot_name = "bot"
imitator_name = "Timur Mukhtarov"
DEFAULT_SYSTEM_PROMPT = "Ты русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им. \n"
DEFAULT_CHAT_PROMT = "Ты имитируешь чат. Отвечай как: "
logs_dir = Config.TRAINING_LOGS_PATH

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
        history.add_message(chat_id, bot_name, phrase)
        bot.send_message(chat_id, phrase)
    except Exception as e:
        bot.send_message(message, f"Ошибка при получении фразы: {str(e)}")


#@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text
        chat_id = message.chat.id

        if not message.from_user.is_bot:
            history.add_message(chat_id, get_fio(message), user_message)
        
        if "@ochen_hueviy_bot" not in user_message:
            return 
        
        discusion = history.get_formatted_history(chat_id)
        prompt = DEFAULT_SYSTEM_PROMPT + discusion

        loggin_promt(prompt)
        
        output = chat_model_mistral.generate_saiga(prompt)
        history.add_message(chat_id, bot_name, output)
        bot.reply_to(message, output)
        
    except Exception as e:
        bot.reply_to(message, f"Ой произошла ошибка: {str(e)}")        

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        chat_id = message.chat.id
        user_message = message.text

        if not message.from_user.is_bot:
            history.add_message(chat_id, get_fio(message), user_message)
        
        if "@ochen_hueviy_bot" not in user_message:
            return 
        
        #discusion = history.get_formatted_history(chat_id)
        
        #prompt = f"{DEFAULT_CHAT_PROMT} [USER:{imitator_name}]. \n Контекст: {user_message}"

        loggin_promt(user_message)

        # Генерируем ответ
        output = chat_model.generate(user_message)

        # Добавляем ответ в историю и отправляем
        history.add_message(chat_id, bot_name, output)
        bot.reply_to(message, output)
        
    except Exception as e:
        bot.reply_to(message, f"Ой произошла ошибка: {str(e)}")

def get_fio(message):
    return f"{message.from_user.first_name} {message.from_user.last_name}"

def loggin_promt(prompt):
    with open(f"{logs_dir}/prompts.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} | {prompt}\n{'-'*40}\n")       

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()