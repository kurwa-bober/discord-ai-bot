"""
config.py - Модуль конфигурации и вспомогательных утилит
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional, List
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv


@dataclass
class BotConfig:
    """
    Класс для хранения и управления конфигурацией бота.
    Загружает переменные из .env файла.
    """
    discord_token: Optional[str] = None
    ai_api_key: Optional[str] = None
    llm_model: str = 'openai/gpt-oss-20b'
    api_url: str = "https://llm.chutes.ai/v1/chat/completions"
    max_history_messages: int = 15
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 60

    def __post_init__(self):
        load_dotenv()
        self.discord_token = os.getenv("DISCORD_BOT_TOKEN")
        self.ai_api_key = os.getenv("AI_PROVIDER_API_KEY")
        self.llm_model = os.getenv("LLM_MODEL", self.llm_model)
        self.api_url = os.getenv("API_URL", self.api_url)

        # Загружаем числовые параметры с проверкой
        try:
            self.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", str(self.max_history_messages)))
            self.max_tokens = int(os.getenv("MAX_TOKENS", str(self.max_tokens)))
            self.temperature = float(os.getenv("TEMPERATURE", str(self.temperature)))
            self.timeout = int(os.getenv("TIMEOUT", str(self.timeout)))
        except ValueError as e:
            logging.warning(f"Ошибка при загрузке числовых параметров из .env: {e}")

    def validate(self) -> None:
        """Проверяет наличие обязательных переменных окружения."""
        if not self.discord_token:
            raise ValueError("Переменная DISCORD_BOT_TOKEN не установлена в .env файле.")
        if not self.ai_api_key:
            raise ValueError("Переменная AI_PROVIDER_API_KEY не установлена в .env файле.")


def setup_logging() -> None:
    """Настройка системы логирования с улучшенным форматированием."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)8s] [%(name)20s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Очищаем существующие обработчики, чтобы избежать дублирования
    root_logger.handlers.clear()

    # Обработчик для записи в файл с ротацией
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "bot.log"),
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)

    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Настройка уровней для внешних библиотек
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


class MessageUtils:
    """Утилиты для работы с сообщениями."""

    @staticmethod
    def split_message(text: str, max_length: int = 2000) -> List[str]:
        """Разбивает длинный текст на части с улучшенной логикой."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        remaining_text = text

        while len(remaining_text) > max_length:
            # Ищем лучшее место для разбивки
            split_pos = -1

            # Приоритет разбивки: абзац -> предложение -> слово
            for delimiter in ['\n\n', '\n', '. ', '! ', '? ', ', ', ' ']:
                pos = remaining_text.rfind(delimiter, 0, max_length)
                if pos != -1:
                    split_pos = pos + len(delimiter)
                    break

            # Если не нашли подходящее место, режем принудительно
            if split_pos == -1:
                split_pos = max_length

            chunk = remaining_text[:split_pos].strip()
            if chunk:
                chunks.append(chunk)

            remaining_text = remaining_text[split_pos:].strip()

        if remaining_text:
            chunks.append(remaining_text)

        return chunks

    @staticmethod
    def get_system_prompt() -> dict:
        """Возвращает системный промпт для LLM."""
        return {
            "role": "system",
            "content": (
                "Ты — ИИ-ассистент в Discord. Твоя задача — поддерживать осмысленный и дружелюбный диалог. "
                "Отвечай на том же языке, на котором к тебе обращаются. "
                "В истории диалога сообщения от пользователей начинаются с их ника. "
                "Стимулируй размышления и задавай встречные вопросы для поддержания диалога. "
                "Будь полезным, информативным и дружелюбным."
            )
        }