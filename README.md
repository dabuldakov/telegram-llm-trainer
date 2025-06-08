# telegram-llm-trainer

To start:

brew install python3
git clone https://github.com/dabuldakov/telegram-llm-trainer.git
pip install -r requirements.txt
create file .env with tokens
TELEGRAM_BOT_TOKEN=your token...
HUGGINGFACE_TOKEN=your token...

Application flow:

1. Prepare dataset from telegram chat history logs

python3 prepare/prepare_data.py

2. Train dataset on LLM model

python3 -m training.train

3. Start Telegram bot

python3 -m bot.bot


Additional settings:

training logs

tensorboard --logdir=outputs/runs