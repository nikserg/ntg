import asyncio
import json
import logging
import os

from aiohttp import web

import collecting_feedback
import invite
import limits
import migrations
import start
from config import print_config, EXCLUDE_WORDS
from llm import run_llm
from messages import save_message
from qdrant import save_message_to_qdrant, init_qdrant_collection
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
            return answer(await collecting_feedback.handle_feedback(chat_id, user_input))

        if user_input.startswith("/start"):
            text, image = await start.handle_command(chat_id, user_input)
            return answer(text, image)

        if user_input == "/feedback":
            return answer(collecting_feedback.handle_command(chat_id))

        if user_input == "/invite":
            return answer(await invite.handle_command(chat_id))

        if user_input == "/limit":
            return answer(await limits.handle_command(chat_id))

        # Проверяем, не превышен ли лимит сообщений
        limit_exceeded, limit_exceeded_message = await limits.is_limit_exceeded(chat_id)
        if limit_exceeded:
            return answer(limit_exceeded_message)

        # Лимит не превышен, формируем ответ пользователю из LLM

        # Очищаем входящее сообщение от лишних символов и сохраняем его в БД
        cleaned_input = remove_newlines(user_input)
        cleaned_input = exclude_words_from_input(cleaned_input, EXCLUDE_WORDS)
        await save_message(chat_id, cleaned_input, "user")
        # Добавляем сообщение пользователя в Qdrant
        await save_message_to_qdrant(chat_id, cleaned_input, "user")

        # Отправляем запрос к LLM
        reply = await run_llm(chat_id, cleaned_input)
        reply_without_newlines = remove_newlines(reply)
        # Добавляем ответ бота в векторное хранилище, но без новых строк
        await save_message_to_qdrant(chat_id, reply_without_newlines, "assistant")
        # Сохраняем ответ в БД (c новыми строками, т.к. это важно для контекста)
        await save_message(chat_id, reply, "assistant")
        logging.info(f"Ответ пользователю {chat_id}: {reply}")
        return answer(reply)
    except Exception as e:
        # Вывести в лог ошибку и стектрейс
        logging.error(f"Ошибка при обработке /internal: {str(e)}")
        return answer(f"Ошибка: {str(e)}", status=500)


def answer(text, image=None, status=200):
    """Формирует ответ для бота в формате JSON."""
    data = {"text": text}
    if image:
        data["image"] = image
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        status=status,
        content_type='application/json'
    )


# При старте приложения выполняем инициализацию Qdrant и миграции
async def on_startup(app):
    # Инициализация Qdrant
    await asyncio.get_event_loop().run_in_executor(None, init_qdrant_collection)
    # Применяем миграции
    await migrations.apply_migrations()


# Создание и запуск aiohttp-приложения
app = web.Application()
app.on_startup.append(on_startup)
app.router.add_post('/internal', handle_internal_request)

if __name__ == "__main__":
    web.run_app(app, host="::", port=int(os.getenv("PORT", 8080)))
