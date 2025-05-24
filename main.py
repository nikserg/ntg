import asyncio
import contextlib
import logging
import os
import re
# === LOAD ENV ===
# Загружаем переменные окружения из .env файла, если он существует
from pathlib import Path

import aiohttp
import tiktoken
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import setup_application, SimpleRequestHandler
from aiohttp import web
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# === НАСТРОЙКА ЛОГИРОВАНИЯ ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === КОНФИГУРАЦИЯ ===
# Считывание всех параметров из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "")
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 4096))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 2500))
MAX_HISTORY_SIZE = int(os.getenv("MAX_HISTORY_SIZE", 1000))
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
CHARACTER_CARD = os.getenv("CHARACTER_CARD", "Имя: Ника\nЛичность: Доброжелательная, отзывчивая, умная\n")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE", "")
WEBHOOK_URL = WEBHOOK_BASE + WEBHOOK_PATH if WEBHOOK_BASE else None
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты виртуальный помощник, который помогает пользователям с их вопросами и задачами. Ты дружелюбный и отзывчивый.")
FIRST_MESSAGE = os.getenv("FIRST_MESSAGE", "Привет, я Ника! А тебя как зовут?")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MIN_P = float(os.getenv("MIN_P", 0))
TOP_P = float(os.getenv("TOP_P", 1))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", 1.1))
REPLY_MAX_TOKENS = int(os.getenv("REPLY_MAX_TOKENS", 400))

# Выводим параметры в лог
logging.info(f"Настройки бота:\n"
             f"TELEGRAM_TOKEN: {TELEGRAM_TOKEN}\n"
             f"RUNPOD_ENDPOINT: {RUNPOD_ENDPOINT}\n"
             f"CONTEXT_LENGTH: {CONTEXT_LENGTH}\n"
             f"CONTEXT_TOKEN_LIMIT: {CONTEXT_TOKEN_LIMIT}\n"
             f"MAX_HISTORY_SIZE: {MAX_HISTORY_SIZE}\n"
             f"CHARACTER_CARD: {CHARACTER_CARD}\n"
             f"WEBHOOK_PATH: {WEBHOOK_PATH}\n"
             f"WEBHOOK_BASE: {WEBHOOK_BASE}\n"
             f"WEBHOOK_URL: {WEBHOOK_URL}\n"
             f"SYSTEM_PROMPT: {SYSTEM_PROMPT}\n"
             f"TEMPERATURE: {TEMPERATURE}\n"
             f"MIN_P: {MIN_P}\n"
             f"TOP_P: {TOP_P}\n"
             f"REPEAT_PENALTY: {REPEAT_PENALTY}\n"
             f"REPLY_MAX_TOKENS: {REPLY_MAX_TOKENS}\n"
             f"RUNPOD_API_KEY: {RUNPOD_API_KEY}\n"
             )

# Комбинированный промпт для LLM
full_system_prompt = f"{SYSTEM_PROMPT}\n{CHARACTER_CARD}"

# === ИНИЦИАЛИЗАЦИЯ ===
# Создаём бота только если есть токен
if TELEGRAM_TOKEN:
    bot = Bot(token=TELEGRAM_TOKEN)
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
else:
    logging.info("TELEGRAM_TOKEN не указан. Бот не будет запущен.")
    bot = None
    storage = None
    dp = None
tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Внутренние хранилища
vector_store = {}
vector_embeddings = {}
chat_history = {}


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
async def keep_typing(chat_id: int, interval: float = 4.0):
    """Периодически отправляет статус 'печатает' в чат."""
    while True:
        await bot.send_chat_action(chat_id, "typing")
        await asyncio.sleep(interval)


@contextlib.asynccontextmanager
async def typing_action(chat_id: int):
    """Контекстный менеджер, поддерживающий статус 'печатает' активным."""
    task = asyncio.create_task(keep_typing(chat_id))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

def remove_newlines(text):
    """Удаляет переносы строк из текста, заменяя их на пробелы"""
    return text.replace("\n", " ")

def trim_incomplete_sentence(text):
    # Находит последнее завершённое предложение
    match = re.search(r'([.!?…\]])[^.!?…\]]*$', text)
    if match:
        end = match.end(1)
        return text[:end].strip()
    return text.strip()


def truncate_history(messages, max_tokens):
    """
    Обрезает историю сообщений, чтобы оставить максимальное
    количество последних сообщений в рамках max_tokens.
    """
    if not messages:
        return []

    # Если все сообщения помещаются в лимит
    total_tokens = sum(len(tokenizer.encode(msg)) for msg in messages)
    if total_tokens <= max_tokens:
        return messages

    # Проверяем, помещается ли последнее сообщение
    last_msg_tokens = len(tokenizer.encode(messages[-1]))
    if last_msg_tokens > max_tokens:
        return []  # Если последнее сообщение не помещается, возвращаем пустой список

    # Собираем сообщения с конца, до достижения лимита
    result = []
    tokens_used = 0

    for msg in reversed(messages):
        msg_tokens = len(tokenizer.encode(msg))
        if tokens_used + msg_tokens <= max_tokens:
            result.insert(0, msg)  # Вставляем в начало для сохранения порядка
            tokens_used += msg_tokens
        else:
            break  # Прекращаем добавлять сообщения, если не помещаются

    return result


def embed_text(text):
    # Генерирует эмбеддинг текста
    return embedder.encode([text], show_progress_bar=False)[0].astype("float32")


def find_similar(text, chat_id, current_context=None, top_k=3):
    # Находит похожие сообщения в векторном хранилище, исключая те, что уже есть в контексте
    if chat_id not in vector_embeddings or not vector_embeddings[chat_id]:
        return []
    if len(vector_embeddings[chat_id]) < 1:
        return []

    # Установим current_context в пустой список, если он не передан
    if current_context is None:
        current_context = []

    vec = embed_text(text).reshape(1, -1)
    n_neighbors = min(top_k * 3, len(vector_embeddings[chat_id]))  # Берем больше соседей с запасом для фильтрации

    index = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='cosine')
    index.fit(vector_embeddings[chat_id])
    distances, indices = index.kneighbors(vec, n_neighbors=n_neighbors)

    # Фильтрация результатов: исключаем сообщения, которые уже есть в контексте
    similar_msgs = []
    for i in indices[0]:
        candidate = vector_store[chat_id][i]
        if candidate not in current_context:
            similar_msgs.append(candidate)
            if len(similar_msgs) >= top_k:
                break

    return similar_msgs


async def run_llm(prompt):
    payload = {
        "input": {
            "prompt": prompt,
            "params": {
                "system_prompt": full_system_prompt,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "min_p": MIN_P,
                "repeat_penalty": REPEAT_PENALTY,
                "max_tokens": REPLY_MAX_TOKENS
            }
        }
    }
    logging.info(f"Отправка запроса к LLM: {payload}")
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            async with session.post(RUNPOD_ENDPOINT + "/runsync", json=payload, headers=headers, timeout=240) as response:
                response.raise_for_status()
                data = await response.json()
                # Проверяем, не вернулся ли статус IN_PROGRESS
                if data.get("status") in ["IN_PROGRESS", "IN_QUEUE"] and data.get("id"):
                    task_id = data.get("id")
                    logging.info(f"Получен статус {data.get('status')}, id задачи: {task_id}")

                    # Ждем завершения задачи
                    max_attempts = 120
                    attempts = 0

                    while attempts < max_attempts:
                        await asyncio.sleep(1)
                        attempts += 1

                        async with session.get(f"{RUNPOD_ENDPOINT}/status/{task_id}",
                                               headers=headers) as status_response:
                            status_data = await status_response.json()
                            status = status_data.get("status")

                            if status == "COMPLETED":
                                output = status_data.get("output", {})
                                text = output.get("text")
                                if text is None:
                                    logging.error(f"Ответ LLM без текста: {status_data}")
                                    return "[Ой! Кажется, у меня техническая проблема под кодовым именем Клубничка]"
                                return text

                            elif status == "FAILED":
                                logging.info(f"Задача завершилась с ошибкой: {status_data}")
                                return "[Ой! Кажется, у меня техническая проблема под кодовым именем Апельсин]"

                            # Иначе продолжаем ожидание

                    logging.error(f"Превышено время ожидания для задачи {task_id}")
                    return "[Ой! Кажется, у меня техническая проблема под кодовым именем Лимон]"

                # Обычная обработка для случая немедленного ответа
                output = data.get("output", {})
                text = output.get("text")
                if text is None:
                    logging.error(f"Ответ LLM без текста: {data}")
                    return "[Ой! Кажется, у меня техническая проблема под кодовым именем Клубничка]"
                return text
    except Exception as e:
        logging.error(f"Ошибка при обращении к RunPod: {e}")
        return "[Ой! Кажется, у меня техническая проблема под кодовым именем Персик]"

def build_prompt(memories, history):
    # Формирует финальный промпт для LLM
    prompt_parts = [
        "Предыдущие события:\n",
        "\n".join(memories).strip() if memories else "нет",
        "\n***\n",
        "\n".join(history).strip(),
        "Ника:"
    ]
    return "\n".join(prompt_parts)

# === ОБРАБОТКА СООБЩЕНИЙ ===
@dp.message()
async def handle_message(message: Message):
    # Выходим, если бот не инициализирован (для тестов)
    if not bot:
        return
    logging.info(f"Входящее сообщение от {message.chat.id}: {message.text}")
    chat_id = message.chat.id
    user_input = message.text.strip()

    # Начинаем поддерживать статус "печатает..." сразу после получения сообщения
    async with typing_action(chat_id):
        if user_input == "/start":
            logging.info(f"Команда /start от {chat_id}")
            first_message = FIRST_MESSAGE.replace('\\n', '\n')
            chat_history[chat_id] = [f"Ника: {remove_newlines(first_message)}"]
            await message.answer(FIRST_MESSAGE.replace("\\n", "\n"))
            return

        if chat_id not in chat_history:
            first_message = FIRST_MESSAGE.replace('\\n', '\n')
            chat_history[chat_id] = [f"Ника: {remove_newlines(first_message)}"]
        if chat_id not in vector_store:
            vector_store[chat_id] = []
            vector_embeddings[chat_id] = []

        cleaned_input = remove_newlines(user_input)
        cleaned_input_with_name = f"Ты: {cleaned_input}"
        chat_history[chat_id].append(cleaned_input_with_name)
        emb = embed_text(cleaned_input_with_name)
        vector_store[chat_id].append(cleaned_input_with_name)
        vector_embeddings[chat_id].append(emb)

        if len(chat_history[chat_id]) > MAX_HISTORY_SIZE:
            chat_history[chat_id] = chat_history[chat_id][-MAX_HISTORY_SIZE:]
            vector_store[chat_id] = vector_store[chat_id][-MAX_HISTORY_SIZE:]
            vector_embeddings[chat_id] = vector_embeddings[chat_id][-MAX_HISTORY_SIZE:]

        # Сначала получаем историю
        history = truncate_history(chat_history[chat_id], CONTEXT_TOKEN_LIMIT)
        # Затем находим похожие сообщения, исключая те, что уже в контексте
        memories = find_similar(cleaned_input, chat_id, current_context=history)

        prompt = build_prompt(memories, history)
        reply = await run_llm(prompt)
        reply = trim_incomplete_sentence(reply)
        cleaned_reply = remove_newlines(reply)
        reply_with_name = f"Ника: {cleaned_reply}"
        # Добавляем ответ бота в векторное хранилище
        emb_reply = embed_text(reply_with_name)
        vector_store[chat_id].append(reply_with_name)
        vector_embeddings[chat_id].append(emb_reply)

        chat_history[chat_id].append(reply_with_name)
        logging.info(f"Ответ пользователю {chat_id}: {reply}")
        await message.answer(reply)


# === ЗАПУСК WEBHOOK ===
async def on_startup(app):
    if WEBHOOK_URL:
        logging.info(f"Установка вебхука на {WEBHOOK_URL}")
        await bot.set_webhook(WEBHOOK_URL)
    else:
        logging.warning("WEBHOOK_URL не указан. Вебхук не будет установлен.")


async def on_shutdown(app):
    if WEBHOOK_URL:
        await bot.delete_webhook()


# Создание и запуск aiohttp-приложения
app = web.Application()
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if WEBHOOK_URL:
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
