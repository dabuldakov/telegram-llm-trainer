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
    DATA_USER_NAMES = "data/names.txt"
    
    # Model
    MODEL_PATH = "outputs/fine_tuned_model"
    MODEL_PATH_FINISHED_TRAIN = "outputs/v1_individual/fine_tuned_model"
    TRAINING_LOGS_PATH = "logs"
    MAX_CHAT_HISTORY = 5