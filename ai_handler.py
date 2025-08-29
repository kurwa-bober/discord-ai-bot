"""
ai_handler.py - Модуль для работы с LLM API и обработки истории сообщений
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union

import aiohttp
import discord

from config import BotConfig, MessageUtils


class AIHandler:
    """Класс для управления взаимодействием с LLM API."""

    def __init__(self, config: BotConfig):
        self.config = config
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.utils = MessageUtils()

    async def initialize(self) -> None:
        """Инициализация HTTP сессии."""
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "Discord-LLM-Bot/1.0"}
        )
        self.logger.info("AI Handler инициализирован")

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
            self.logger.info("HTTP сессия закрыта")

    async def query_ai(self, messages: List[Dict[str, str]]) -> str:
        """Отправляет асинхронный запрос к LLM API с улучшенной обработкой ошибок."""
        if not self.http_session or self.http_session.closed:
            self.logger.error("HTTP сессия неактивна")
            return "❌ Ошибка: Проблема с сетевым подключением."

        headers = {
            "Authorization": f"Bearer {self.config.ai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }

        try:
            async with self.http_session.post(
                    self.config.api_url,
                    headers=headers,
                    json=payload
            ) as response:
                if response.status == 429:
                    self.logger.warning("Превышен лимит запросов к API")
                    return "⏰ Извините, сервис временно перегружен. Попробуйте через несколько минут."

                response.raise_for_status()
                data = await response.json()

                content = data.get('choices', [{}])[0].get('message', {}).get('content')
                if content:
                    return content.strip()

                self.logger.error(f"Пустой ответ от API: {data}")
                return "🤔 Извините, получен пустой ответ от ИИ."

        except aiohttp.ClientResponseError as e:
            self.logger.error(f"Ошибка API (статус {e.status}): {e.message}")
            if e.status == 401:
                return "🔐 Ошибка авторизации API. Обратитесь к администратору."
            elif e.status == 503:
                return "🔧 Сервис ИИ временно недоступен. Попробуйте позже."
            else:
                return f"⚠️ Ошибка сервиса ИИ (код {e.status}). Попробуйте позже."

        except asyncio.TimeoutError:
            self.logger.error("Превышено время ожидания ответа от API")
            return "⏱️ Извините, ИИ не ответил вовремя. Попробуйте упростить запрос."

        except Exception as e:
            self.logger.exception("Неожиданная ошибка при запросе к ИИ")
            return "💥 Произошла неожиданная ошибка. Попробуйте позже."


class ConversationManager:
    """Класс для управления историей диалога."""

    def __init__(self, config: BotConfig, bot_user: discord.User):
        self.config = config
        self.bot_user = bot_user
        self.logger = logging.getLogger(self.__class__.__name__)
        self.utils = MessageUtils()

    def clean_message_content(self, content: str, display_name: str, role: str) -> str:
        """Очищает содержимое сообщения."""
        # Убираем упоминания бота
        content = content.replace(f'<@!{self.bot_user.id}>', '').replace(f'<@{self.bot_user.id}>', '').strip()

        # Добавляем имя пользователя для контекста
        if role == "user" and content:
            content = f"{display_name}: {content}"

        return content

    async def get_formatted_history(self, message: discord.Message) -> List[Dict[str, str]]:
        """Собирает историю сообщений для обычных упоминаний."""
        history_messages = [
            msg async for msg in message.channel.history(
                limit=self.config.max_history_messages,
                before=message.created_at
            )
        ]

        conversation = []
        for msg in reversed(history_messages):
            role = "assistant" if msg.author == self.bot_user else "user"
            content = self.clean_message_content(msg.content, msg.author.display_name, role)

            if content.strip():  # Игнорируем пустые сообщения
                conversation.append({"role": role, "content": content})

        # Добавляем текущее сообщение
        current_content = self.clean_message_content(
            message.content,
            message.author.display_name,
            "user"
        )
        conversation.append({"role": "user", "content": current_content})

        return [self.utils.get_system_prompt()] + conversation

    async def get_formatted_history_from_channel(
            self,
            channel: Union[discord.TextChannel, discord.Thread],
            user_message: str,
            user: discord.User
    ) -> List[Dict[str, str]]:
        """Собирает историю сообщений для slash команд."""
        history_messages = [
            msg async for msg in channel.history(limit=self.config.max_history_messages)
        ]

        conversation = []
        for msg in reversed(history_messages):
            role = "assistant" if msg.author == self.bot_user else "user"
            content = self.clean_message_content(msg.content, msg.author.display_name, role)

            if content.strip():
                conversation.append({"role": role, "content": content})

        # Добавляем текущее сообщение пользователя
        conversation.append({
            "role": "user",
            "content": f"{user.display_name}: {user_message}"
        })

        return [self.utils.get_system_prompt()] + conversation

    async def get_simple_history(self, user_message: str, user: discord.User) -> List[Dict[str, str]]:
        """Создает простую историю без контекста канала."""
        return [
            self.utils.get_system_prompt(),
            {"role": "user", "content": f"{user.display_name}: {user_message}"}
        ]