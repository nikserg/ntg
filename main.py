import asyncio
import json
import logging
import os

from aiohttp import web

import collecting_feedback
import invite
import limits
import llm
import migrations
import qdrant
import start
from config import print_config, EXCLUDE_WORDS
from db import save_message, get_current_messages
from sanitizer import remove_newlines, exclude_words_from_input

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Выводим параметры в лог
print_config()


async def handle_internal_request(request):
    """Обработчик для внутреннего API эндпойнта."""
    try:
        # Получаем текст запроса напрямую из тела
        data = await request.json()
        user_input = data["text"].strip()
        chat_id = data["chat_id"]
        logging.info(f"Входящее сообщение от {chat_id}: {user_input}")

        # Проверяем, собираем ли отзыв
        if collecting_feedback.is_collecting_feedback(chat_id):
            return json_response({"text": await collecting_feedback.handle_feedback(chat_id, user_input)})

        if user_input.startswith("/start"):
            return json_response({"text": await start.handle_command(chat_id, user_input)})

        if user_input == "/feedback":
            return json_response({"text": collecting_feedback.handle_command(chat_id)})

        if user_input == "/invite":
            return json_response({"text": await invite.handle_command(chat_id)})

        if user_input == "/limit":
            return json_response({"text": await limits.handle_command(chat_id)})

        # Проверяем, не превышен ли лимит сообщений
        limit_exceeded, limit_exceeded_message = await limits.is_limit_exceeded(chat_id)
        if limit_exceeded:
            return json_response({"text": limit_exceeded_message})

        # Лимит не превышен, формируем ответ пользователю из LLM

        # Пользователь отправил сообщение, однако история сообщений для этого чата ещё не создана
        if not await get_current_messages(chat_id):
            await start.add_first_messages_to_db(chat_id)

        # Очищаем входящее сообщение от лишних символов и сохраняем его в БД
        cleaned_input = remove_newlines(user_input)
        cleaned_input = exclude_words_from_input(cleaned_input, EXCLUDE_WORDS)
        await save_message(chat_id, cleaned_input, "user")
        # Добавляем сообщение пользователя в Qdrant
        await qdrant.save_message_to_qdrant(chat_id, cleaned_input, "user")

        # Отправляем запрос к LLM
        reply = await llm.run_llm(chat_id, cleaned_input)
        reply_without_newlines = remove_newlines(reply)
        # Добавляем ответ бота в векторное хранилище, но без новых строк
        await qdrant.save_message_to_qdrant(chat_id, reply_without_newlines, "assistant")
        # Сохраняем ответ в БД (c новыми строками, т.к. это важно для контекста)
        await save_message(chat_id, reply, "assistant")
        logging.info(f"Ответ пользователю {chat_id}: {reply}")
        return json_response({"text": reply})
    except Exception as e:
        logging.error(f"Ошибка при обработке /internal: {e}")
        return json_response({"text": f"Ошибка: {str(e)}"}, 500)


def json_response(data, status=200):
    """Утилита для создания JSON ответа."""
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        status=status,
        content_type='application/json'
    )

# При старте приложения выполняем инициализацию Qdrant и миграции
async def on_startup(app):
    # Инициализация Qdrant
    await asyncio.get_event_loop().run_in_executor(None, qdrant.init_qdrant_collection)
    # Применяем миграции
    await migrations.apply_migrations()


# Создание и запуск aiohttp-приложения
app = web.Application()
app.on_startup.append(on_startup)
app.router.add_post('/internal', handle_internal_request)

if __name__ == "__main__":
    web.run_app(app, host="::", port=int(os.getenv("PORT", 8080)))
