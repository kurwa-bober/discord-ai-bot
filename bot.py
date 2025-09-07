import os
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import aiohttp
import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

AI_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Ты — ИИ-ассистент в Discord. Твоя задача — поддерживать осмысленный и информативный диалог. "
        "Всегда отвечай кратко. "
        "Отвечай на том же языке, на котором к тебе обращаются. "
        # "В истории диалога сохраняется только текст сообщений, без имён пользователей. "
    ),
}

LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

formatter = logging.Formatter("%(asctime)s [%(levelname)8s] [%(name)20s] %(message)s", "%Y-%m-%d %H:%M:%S")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "bot.log"),
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding="utf-8"
)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

class AIClient:
    def __init__(self):
        self.client_timeout = 60
        self.tcp_limit = 30
        self.tcp_limit_per_host = 10
        self.api_url = "https://llm.chutes.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('AI_PROVIDER_API_KEY')}",
            "Content-Type": "application/json",
        }

    def _create_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            limit=self.tcp_limit,
            limit_per_host=self.tcp_limit_per_host,
        )
        timeout = aiohttp.ClientTimeout(total=self.client_timeout)
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "Discord-AI-Bot/1.0"},
        )


    async def query(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": os.getenv("AI_LLM_MODEL"),
            "messages": messages,
            "stream": False,
            "max_tokens": int(os.getenv("AI_MAX_TOKENS")),
            "temperature": float(os.getenv("AI_TEMPERATURE")),
        }
        async with self._create_session() as session:
            response = await session.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = await response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )


class DiscordAIBot(discord.Client):
    def __init__(self, intents: discord.Intents):
        super().__init__(intents=intents)
        self.logger = logging.getLogger("DiscordAIBot")
        self.tree = app_commands.CommandTree(self)
        self.ai = AIClient()

    async def setup_hook(self) -> None:
        await self._register_commands()

    async def _register_commands(self) -> None:
        @self.tree.command(name="help", description="Справка по командам")
        async def cmd_help(interaction: discord.Interaction) -> None:
            embed = discord.Embed(
                title="🤖 Команды",
                description="Вот что я умею:",
                color=discord.Color.green(),
                timestamp=datetime.now(timezone.utc),
            )
            embed.add_field(name="📊 `/status`", value="Показать статус бота", inline=False)
            embed.add_field(name="❓ `/help`", value="Показать справку", inline=False)
            embed.add_field(
                name="📌 Дополнительно",
                value="Можно просто упомянуть меня (@бот), и я отвечу.",
                inline=False,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @self.tree.command(name="status", description="Статус бота")
        async def cmd_status(interaction: discord.Interaction) -> None:
            embed = discord.Embed(
                title="📊 Статус бота",
                color=discord.Color.blue(),
                timestamp=datetime.now(timezone.utc),
            )
            embed.add_field(
                name="🤖 Информация",
                value=f"**Модель:** {os.getenv('AI_LLM_MODEL')}\n"
                f"**Серверов:** {len(self.guilds)}\n"
                f"**Пинг:** {round(self.latency * 1000)}ms",
                inline=False,
            )
            embed.add_field(
                name="⚙️ Настройки",
                value=f"**История:** {os.getenv('AI_DISCORD_MESSAGE_HISTORY')} сообщений\n"
                f"**Max токенов:** {os.getenv('AI_MAX_TOKENS')}\n"
                f"**Температура:** {os.getenv('AI_TEMPERATURE')}\n",
                inline=False,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

    async def on_ready(self) -> None:
        self.logger.info(f"✅ Вошёл как {self.user} (ID: {self.user.id})")
        synced = await self.tree.sync()
        self.logger.info(f"Синхронизировано {len(synced)} команд")

    async def _get_message_history(self, size: int, message: discord.Message) -> list[str]:
        messages: list[str] = []
        async for msg in message.channel.history(limit=size, before=message.created_at, oldest_first=False):
            content = msg.content.strip()
            if content:
                messages.append(content)
        messages.reverse()
        return messages

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user or not self.user.mentioned_in(message):
            return
        await message.add_reaction("⏳")

        try:
            past_messages = await self._get_message_history(
                int(os.getenv("AI_DISCORD_MESSAGE_HISTORY")), message
            )

            query = [AI_SYSTEM_PROMPT]
            query.extend({"role": "user", "content": t} for t in past_messages)
            query.append({"role": "user", "content": message.content.strip()})

            response = await self.ai.query(query)
            await message.reply(response)
            await message.remove_reaction("⏳", self.user)
        except:
            await message.add_reaction("❌")
            raise


def main():
    intents = discord.Intents.default()
    intents.message_content = True
    bot = DiscordAIBot(intents=intents)
    bot.run(os.getenv("DISCORD_BOT_TOKEN"), log_handler=None)


if __name__ == "__main__":
    main()
