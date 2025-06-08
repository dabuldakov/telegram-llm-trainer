import telebot
from bot.request import Request
from bot.chat_model import ChatModel
from config import Config
from bot.chat_history import ChatHistory

# Инициализация
bot = telebot.TeleBot(Config.TELEGRAM_BOT_TOKEN)
chat_model = ChatModel()
history = ChatHistory()

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        chat_id = message.chat.id
        user_message = message.text
        user_name = "Дмитрий Булдаков"
        imitator_name = "Timur Mukhtarov"
        
        # Добавляем сообщение в историю
        history.add_message(chat_id, user_name, user_message)
        
        # Формируем промпт
        discusion = history.get_formatted_history(chat_id)
        prompt = f"Ты имитируешь чат. Отвечай как: {imitator_name} История: {discusion} Текущее сообщение: {user_message}"
        print(f"[PROMPT]: {prompt}")
        request = Request(user=imitator_name, prompt=prompt)
        
        # Генерируем ответ
        bot_response = chat_model.generate(request)
        
        # Добавляем ответ в историю и отправляем
        history.add_message(chat_id, imitator_name, bot_response)
        bot.reply_to(message, bot_response)
        
    except Exception as e:
        bot.reply_to(message, f"Ой произошла ошибка: {str(e)}")

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling()