import os

class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']

    #Huggingface LLM site
    HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

    #statistics wandb
    WANDB_TOKEN= os.environ['WANDB_TOKEN']

    #Dataset from chat history
    DATA_SET_PATH = "data/processed_dataset"
    DATA_USER_NAMES = "data/user.txt"
    
    # Model
    MODEL_PATH = "outputs/fine_tuned_model"
    MODEL_PATH_FINISHED_TRAIN = "outputs/v0/fine_tuned_model"
    TRAINING_LOGS_PATH = "logs"
    MAX_HISTORY = 5  # Количество сообщений в контексте
    MAX_LENGTH = 500  # Максимальная длина ответа
    MAX_CHAT_HISTORY = 10