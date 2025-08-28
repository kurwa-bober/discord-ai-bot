import discord
import requests
import json
import os
import logging
import argparse
from logging.handlers import RotatingFileHandler

DEFAULT_LLM = 'llama-3.1-8b-instant'

# -------- Command-line Arguments -------- #
parser = argparse.ArgumentParser(description="Run AI Discord Bot")
parser.add_argument("-k", "--ai-provider-api-key", required=True, help="AI provider API key")
parser.add_argument("-t", "--discord-bot-token", required=True, help="Discord bot token")
parser.add_argument("-l", "--llm", required=False, help="LLM name", default=DEFAULT_LLM)
args = parser.parse_args()
print(args)

# -------- Logging Setup -------- #
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        RotatingFileHandler("logs/bot.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# -------- Groq API call -------- #
def query_ai(message: list, model: str, response_format: dict = None) -> str:
    response = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        json={
            "model": model,
            "messages": message,
            "response_format": response_format,
        },
        headers={
            'Authorization': f'Bearer {args.ai_provider_api_key}',
            'Content-Type': 'application/json'
        })
    response.raise_for_status()
    content = json.loads(response.text)['choices'][0]['message']['content']
    logging.info(f"Groq API response: {content[:100]}...")
    return content


# -------- Discord Bot -------- #
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    logging.info(f'✅ Logged in as {client.user}')


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if client.user in message.mentions:
        try:
            await message.add_reaction("⏳")
            logging.info(f"User {message.author}: {message.content}")

            messages = [
                {"role": "system", "content": "You are AI assistant. Provide concise answers."},
                {"role": "user", "content": message.content}
            ]

            response = query_ai(messages, args.llm)

            await message.channel.send(response)

            # await message.clear_reaction("⏳")
            await message.add_reaction("✅")
            logging.info(f"Reply sent to {message.author}")

        except Exception:
            # await message.clear_reaction("⏳")
            await message.add_reaction("❌")
            logging.exception("Error handling message")


# Run the bot
client.run(args.discord_bot_token)
