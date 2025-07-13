import os

class Config:
    # Telegram
    TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']

    #Huggingface LLM site
    HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']

    #statistics wandb
    WANDB_TOKEN= os.environ['WANDB_TOKEN']

    #Prepare chat history to dataset
    CHAT_HISTORY_PATH = "data/chat_history.json"
    CHAT_HISTORY_PREPARED_PATH = "data/chat_history_prepared.json"
    TEXT_DATA_FOR_LLM_SAVE_PATH = "data/text_data_for_llm.txt"

    #Dataset from chat history
    DATA_SET_PATH = "data/processed_dataset"
    DATA_USER_NAMES = "data/names.txt"
    
    # Model
    MODEL_PATH = "outputs/fine_tuned_model"
    MODEL_PATH_FINISHED_TRAIN = "outputs/v2/fine_tuned_model"
    TRAINING_LOGS_PATH = "logs"
    MAX_CHAT_HISTORY = 5