# telegram-llm-trainer

To start:

brew install python3
git clone https://github.com/dabuldakov/telegram-llm-trainer.git
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

create file .env with tokens or add secrets on platform like https://datasphere.yandex.cloud/

https://core.telegram.org/bots
TELEGRAM_BOT_TOKEN=your token...

https://huggingface.co/IlyaGusev/saiga_mistral_7b_lora
HUGGINGFACE_TOKEN=your token...

Deploy on service:
Use git clone on notebook yandex cloud
https://datasphere.yandex.cloud/

Application flow:

1. Prepare dataset from telegram chat history logs

!python3 prepare/prepare_data_individual.py

    1.1 For analyze data

    !python3 prepare/analyze_texts_json.py

2. Train dataset on LLM model

!python3 -m training.train_v3_individual

3. Start Telegram bot

!python3 -m bot.bot

4. Looking for training on site
wandb login
enter here login from https://wandb.ai/authorize