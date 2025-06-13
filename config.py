import os

class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']

    #Huggingface LLM site
    HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

    #Dataset from chat history
    DATA_SET_PATH = "data/processed_dataset"
    
    # Model
    MODEL_PATH = "outputs/fine_tuned_model"
    MODEL_PATHS = {
    "123456789": "outputs/fine_tuned_model",
    }
    MAX_HISTORY = 5  # Количество сообщений в контексте
    MAX_LENGTH = 500  # Максимальная длина ответа
    MAX_CHAT_HISTORY = 10