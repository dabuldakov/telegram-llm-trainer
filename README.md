# telegram-llm-trainer

To start:

brew install python3
git clone https://github.com/dabuldakov/telegram-llm-trainer.git
pip install -r requirements.txt

create file .env with tokens or add secrets on platform like https://datasphere.yandex.cloud/

https://core.telegram.org/bots
TELEGRAM_BOT_TOKEN=your token...

https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora
HUGGINGFACE_TOKEN=your token...

Deploy on service:
https://datasphere.yandex.cloud/

Application flow:

1. Prepare dataset from telegram chat history logs

python3 prepare/prepare_data.py

2. Train dataset on LLM model

python3 -m training.train

3. Start Telegram bot

python3 -m bot.bot

4. Looking for training on site
wandb login
enter here login from https://wandb.ai/authorize