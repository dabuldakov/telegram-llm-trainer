import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

    #Huggingface LLM site
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

    #Dataset from chat history
    DATA_SET_PATH = "data/processed_dataset"
    
    # Model
    MODEL_PATH = "outputs/fine_tuned_model"
    MAX_HISTORY = 5  # Количество сообщений в контексте
    MAX_LENGTH = 500  # Максимальная длина ответа