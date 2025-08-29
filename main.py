"""
main.py - Основной модуль Discord бота с поддержкой LLM
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict

import discord
from discord import app_commands

from config import BotConfig, setup_logging, MessageUtils
from ai_handler import AIHandler, ConversationManager


class DiscordLLMBot(discord.Client):
    """
    Основной класс Discord бота с поддержкой LLM и slash команд.
    """

    def __init__(self, config: BotConfig, intents: discord.Intents):
        super().__init__(intents=intents)
        self.config = config
        self.ai_handler = AIHandler(config)
        self.conversation_manager: Optional[ConversationManager] = None
        self.typing_tasks: Dict[int, asyncio.Task] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.utils = MessageUtils()

        # Инициализация дерева команд
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        """Настройка бота после подключения."""
        # Инициализируем AI handler
        await self.ai_handler.initialize()

        # Инициализируем менеджер диалогов (требует self.user)
        self.conversation_manager = ConversationManager(self.config, self.user)

        # Регистрируем slash команды
        await self._register_commands()

        self.logger.info("Настройка бота завершена.")

    async def _register_commands(self) -> None:
        """Регистрация slash команд."""

        @self.tree.command(name="ask", description="Задать вопрос или начать беседу с ИИ-ассистентом")
        @app_commands.describe(
            message="Ваш вопрос или сообщение для ИИ",
            include_history="Учитывать историю сообщений в канале (по умолчанию: да)"
        )
        async def ask_command(
                interaction: discord.Interaction,
                message: str,
                include_history: bool = True
        ) -> None:
            await self._handle_ai_interaction(interaction, message, include_history)

        @self.tree.command(name="help", description="Показать справку по командам бота")
        async def help_command(interaction: discord.Interaction) -> None:
            await self._handle_help_command(interaction)

        @self.tree.command(name="status", description="Показать статус бота и текущие настройки")
        async def status_command(interaction: discord.Interaction) -> None:
            await self._handle_status_command(interaction)

    async def _handle_help_command(self, interaction: discord.Interaction) -> None:
        """Обработка команды help."""
        embed = discord.Embed(
            title="🤖 Справка по командам бота",
            description="Вот что я умею:",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc)
        )

        commands_info = [
            ("📝 `/ask <сообщение>`", "Задать вопрос или начать беседу с ИИ-ассистентом"),
            ("📊 `/status`", "Показать статус бота и текущие настройки"),
            ("❓ `/help`", "Показать эту справку")
        ]

        for name, description in commands_info:
            embed.add_field(name=name, value=description, inline=False)

        embed.add_field(
            name="📌 Дополнительно",
            value="Вы также можете упомянуть меня в сообщении (@бот), и я отвечу!",
            inline=False
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _handle_status_command(self, interaction: discord.Interaction) -> None:
        """Обработка команды status."""
        embed = discord.Embed(
            title="📊 Статус бота",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc)
        )

        # Информация о боте
        embed.add_field(
            name="🤖 Информация",
            value=f"**Модель:** {self.config.llm_model}\n"
                  f"**Серверов:** {len(self.guilds)}\n"
                  f"**Пинг:** {round(self.latency * 1000)}ms",
            inline=False
        )

        # Настройки
        embed.add_field(
            name="⚙️ Настройки",
            value=f"**История:** {self.config.max_history_messages} сообщений\n"
                  f"**Max токенов:** {self.config.max_tokens}\n"
                  f"**Температура:** {self.config.temperature}",
            inline=False
        )

        # Статус сессии
        session_status = "✅ Активна" if (self.ai_handler.http_session and
                                          not self.ai_handler.http_session.closed) else "❌ Неактивна"
        embed.add_field(
            name="🌐 HTTP сессия",
            value=session_status,
            inline=False
        )

        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _handle_ai_interaction(
            self,
            interaction: discord.Interaction,
            user_message: str,
            include_history: bool = True
    ) -> None:
        """Обработка взаимодействия с ИИ через slash команды."""
        self.logger.info(
            f"Получена команда от {interaction.user} в канале {interaction.channel.name}: "
            f"'{user_message[:50]}...'"
        )

        # Сразу отвечаем, чтобы избежать таймаута
        try:
            await interaction.response.defer()
        except discord.NotFound:
            self.logger.warning("Interaction уже истек, пытаемся ответить напрямую")
            try:
                await interaction.followup.send("⏳ Обрабатываю ваш запрос...")
            except Exception:
                self.logger.error("Не удалось отправить начальный ответ")
                return

        try:
            # Создаем задачу для индикатора печати
            typing_task = asyncio.create_task(self._typing_for_interaction(interaction))

            try:
                # Получаем историю диалога
                if include_history:
                    history = await self.conversation_manager.get_formatted_history_from_channel(
                        interaction.channel,
                        user_message,
                        interaction.user
                    )
                else:
                    history = await self.conversation_manager.get_simple_history(
                        user_message,
                        interaction.user
                    )

                # Запрос к LLM
                ai_response = await self.ai_handler.query_ai(history)

                # Отправляем ответ (разбиваем если длинный)
                chunks = self.utils.split_message(ai_response)

                for i, chunk in enumerate(chunks):
                    if i == 0:
                        try:
                            await interaction.followup.send(chunk)
                        except discord.NotFound:
                            # Если interaction исчез, отправляем в канал
                            await interaction.channel.send(f"{interaction.user.mention} {chunk}")
                    else:
                        try:
                            await interaction.followup.send(chunk)
                        except discord.NotFound:
                            await interaction.channel.send(chunk)

                self.logger.info(f"Успешно отправлен ответ пользователю {interaction.user}")

            finally:
                # Останавливаем индикатор печати
                typing_task.cancel()
                try:
                    await typing_task
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            self.logger.exception(f"Ошибка при обработке команды от {interaction.user}")
            try:
                error_embed = discord.Embed(
                    title="❌ Ошибка",
                    description="Произошла ошибка при обработке вашего запроса. Попробуйте позже.",
                    color=discord.Color.red()
                )
                try:
                    await interaction.followup.send(embed=error_embed)
                except discord.NotFound:
                    await interaction.channel.send(
                        f"{interaction.user.mention} Произошла ошибка при обработке запроса.")
            except Exception:
                self.logger.exception("Не удалось отправить сообщение об ошибке")

    async def on_ready(self) -> None:
        """Вызывается, когда бот успешно подключился к Discord."""
        self.logger.info(f"✅ Успешный вход: {self.user} (ID: {self.user.id})")
        self.logger.info(f"Бот активен на {len(self.guilds)} сервере(ах)")
        self.logger.info(f"Используемая модель LLM: {self.config.llm_model}")

        # Синхронизируем команды
        try:
            synced = await self.tree.sync()
            self.logger.info(f"Синхронизировано {len(synced)} slash команд")
        except Exception as e:
            self.logger.error(f"Ошибка при синхронизации команд: {e}")

    async def on_app_command_error(
            self,
            interaction: discord.Interaction,
            error: app_commands.AppCommandError
    ) -> None:
        """Обработка ошибок в slash командах."""
        self.logger.error(
            f"Ошибка в команде {interaction.command.name if interaction.command else 'неизвестной'}: {error}")

        # Пытаемся отправить понятное сообщение пользователю
        error_message = "❌ Произошла ошибка при выполнении команды."

        if isinstance(error, app_commands.CommandOnCooldown):
            error_message = f"⏰ Команда на перезарядке. Попробуйте через {error.retry_after:.1f} секунд."
        elif isinstance(error, app_commands.MissingPermissions):
            error_message = "🔒 У вас нет прав для выполнения этой команды."
        elif isinstance(error, app_commands.BotMissingPermissions):
            error_message = "🤖 У бота нет необходимых прав для выполнения команды."

        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(error_message, ephemeral=True)
            else:
                await interaction.followup.send(error_message, ephemeral=True)
        except Exception:
            # Если не можем ответить через interaction, отправляем в канал
            try:
                await interaction.channel.send(f"{interaction.user.mention} {error_message}")
            except Exception:
                self.logger.error("Не удалось отправить сообщение об ошибке пользователю")

    async def on_disconnect(self) -> None:
        """Вызывается при отключении бота."""
        self.logger.warning("Бот отключается. Закрытие сессии...")
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Очистка ресурсов."""
        # Отменяем все активные задачи печати
        for task in self.typing_tasks.values():
            task.cancel()
        self.typing_tasks.clear()

        # Закрываем HTTP сессию
        await self.ai_handler.cleanup()

    async def on_message(self, message: discord.Message) -> None:
        """Обрабатывает входящие сообщения (только упоминания)."""
        # Игнорируем пинги от самого себя
        if message.author == self.user:
            return

        # Проверяем упоминание бота
        if not self.user.mentioned_in(message):
            return

        channel_id = message.channel.id
        self.logger.info(f"Получено упоминание от {message.author} в канале {message.channel.name}")

        # Запускаем индикатор печати и добавляем реакцию
        self.typing_tasks[channel_id] = asyncio.create_task(self._typing_indicator(message.channel))

        try:
            await message.add_reaction("⏳")
        except discord.Forbidden:
            self.logger.warning(f"Нет прав для добавления реакций в канале {message.channel.name}")

        try:
            # Получаем историю диалога
            history = await self.conversation_manager.get_formatted_history(message)

            # Запрос к LLM
            ai_response = await self.ai_handler.query_ai(history)

            # Отправляем ответ по частям если нужно
            for chunk in self.utils.split_message(ai_response):
                await message.reply(chunk)

            await message.add_reaction("✅")

        except Exception as e:
            self.logger.exception("Ошибка при обработке упоминания")
            try:
                await message.add_reaction("❌")
            except discord.Forbidden:
                pass
        finally:
            # Останавливаем индикатор печати
            if channel_id in self.typing_tasks:
                self.typing_tasks[channel_id].cancel()
                del self.typing_tasks[channel_id]

            # Удаляем реакцию ожидания
            try:
                await message.remove_reaction("⏳", self.user)
            except (discord.Forbidden, discord.NotFound):
                pass

    async def _typing_indicator(self, channel: discord.abc.Messageable) -> None:
        """Показывает индикатор печати в канале."""
        try:
            async with channel.typing():
                await asyncio.sleep(float('inf'))
        except asyncio.CancelledError:
            pass
        except discord.Forbidden:
            self.logger.warning(
                f"Нет прав для индикатора печати в канале {getattr(channel, 'name', channel.id)}"
            )
        except Exception as e:
            self.logger.warning(f"Ошибка в задаче индикатора печати: {e}")

    async def _typing_for_interaction(self, interaction: discord.Interaction) -> None:
        """Индикатор печати для slash команд."""
        try:
            # Проверяем, что канал доступен
            if not hasattr(interaction, 'channel') or not interaction.channel:
                return

            while True:
                try:
                    async with interaction.channel.typing():
                        await asyncio.sleep(8)  # Discord typing indicator длится ~10 секунд
                except discord.Forbidden:
                    self.logger.warning(f"Нет прав для typing в канале {interaction.channel}")
                    break
                except Exception:
                    # Если произошла ошибка, ждем и пробуем снова
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.warning(f"Ошибка в индикаторе печати для interaction: {e}")

    def start_bot(self) -> None:
        """Основной блок для начала запуска."""
        try:
            self.config.validate()
            self.logger.info("Запуск бота...")
            self.run(self.config.discord_token, log_handler=None)  # Отключаем встроенное логирование Discord
        except discord.LoginFailure:
            self.logger.critical("Ошибка авторизации: неверный токен Discord")
        except ValueError as e:
            self.logger.critical(f"Ошибка конфигурации: {e}")
        except KeyboardInterrupt:
            self.logger.info("Получен сигнал прерывания. Завершение работы...")
        except Exception as e:
            self.logger.critical("Критическая ошибка при запуске бота", exc_info=True)
        finally:
            # Принудительная очистка
            asyncio.run(self._cleanup())


def main():
    """Запуск """
    # Настраиваем логирование
    setup_logging()
    logger = logging.getLogger("main")

    # Настройка прав бота
    intents = discord.Intents.default()
    intents.message_content = True
    intents.messages = True
    intents.guilds = True

    # Запуск бота
    try:
        config = BotConfig()
        bot = DiscordLLMBot(config=config, intents=intents)
        logger.info("Конфигурация загружена, запуск бота...")
        bot.start_bot()
    except Exception as e:
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())