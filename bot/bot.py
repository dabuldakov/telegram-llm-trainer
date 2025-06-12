import telebot
from bot.chat_model_saiga_mistral import ChatModelSaigaMistral, generate_saiga
from bot.request import Request
#from bot.chat_model import ChatModel
from config import Config
from bot.chat_history import ChatHistory

# Инициализация
bot = telebot.TeleBot(Config.TELEGRAM_BOT_TOKEN)
#chat_model = ChatModel()
chat_model_mistral = ChatModelSaigaMistral()
history = ChatHistory()
bot_name = "HuinyaBot"
imitator_name = "Timur Mukhtarov"

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


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text
        
        chat_model_mistral.add_user_message(user_message)
        # Проверяем наличие упоминания бота
        if "@ochen_hueviy_bot" not in user_message:
            return 
        
        # Генерируем ответ
        prompt = chat_model_mistral.get_prompt()
        output = generate_saiga(prompt)
        chat_model_mistral.add_bot_message(output)
        # Отправляем
        bot.reply_to(message, output)
        
    except Exception as e:
        bot.reply_to(message, f"Ой произошла ошибка: {str(e)}")      

#@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        chat_id = message.chat.id
        user_message = message.text
        
        # Добавляем сообщение в историю
        fio = f"{message.from_user.first_name} {message.from_user.last_name}"
        

        # Проверяем наличие упоминания бота
        if "@ochen_hueviy_bot" not in user_message:
            history.add_message(chat_id, fio, user_message)
            return 
        
        # Формируем промпт
        discusion = history.get_formatted_history(chat_id)
        prompt = f"Ты имитируешь чат. Отвечай как: {imitator_name}. {user_message} \n Контекст: {discusion}"
        print(f"[PROMPT]: {prompt}")
        request = Request(user=imitator_name, prompt=prompt)
        
        # Генерируем ответ
        #bot_response = chat_model.generate(request, chat_id)

        # Добавляем ответ в историю и отправляем
        #history.add_message(chat_id, bot_name, bot_response)
        #bot.reply_to(message, bot_response)
        
    except Exception as e:
        bot.reply_to(message, f"Ой произошла ошибка: {str(e)}")

import random

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()