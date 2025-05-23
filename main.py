import logging
import os
import time
import aiohttp
import asyncio

# === LOAD ENV ===
# Загружаем переменные окружения из .env файла, если он существует
from pathlib import Path

import tiktoken
from aiogram import Bot, Dispatcher, types
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from aiogram.enums import ChatAction

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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 4096))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 2500))
MAX_HISTORY_SIZE = int(os.getenv("MAX_HISTORY_SIZE", 1000))
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
CHARACTER_CARD = os.getenv("CHARACTER_CARD",
                           "Name: Ника\nPersonality: Игривая, кокетливая, милая, немного дерзкая\nAppearance: Розовые волосы, пронзительные зелёные глаза, кружевной чокер")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
WEBHOOK_URL = WEBHOOK_BASE + WEBHOOK_PATH if WEBHOOK_BASE else None
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
                          "Напиши один ответ от лица Ники. Как минимум один параграф, максимум четыре. Подробно и с погружением описывай все детали о действиях, эмоциях и окружении Ники. Старайся использовать разнообразный язык.")
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
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Внутренние хранилища
vector_store = {}
vector_embeddings = {}
chat_history = {}


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===
def truncate_history(messages, max_tokens):
    # Сокращает историю сообщений по лимиту токенов
    total = 0
    kept = []
    for msg in reversed(messages):
        tokens = len(tokenizer.encode(msg))
        if total + tokens > max_tokens:
            break
        kept.append(msg)
        total += tokens
    return list(reversed(kept))


def embed_text(text):
    # Генерирует эмбеддинг текста
    return embedder.encode([text])[0].astype("float32")


def find_similar(text, chat_id, top_k=3):
    # Находит похожие сообщения в векторном хранилище
    if chat_id not in vector_embeddings or not vector_embeddings[chat_id]:
        return []
    if len(vector_embeddings[chat_id]) < 1:
        return []
    vec = embed_text(text).reshape(1, -1)
    n_neighbors = min(top_k, len(vector_embeddings[chat_id]))
    index = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', metric='euclidean')
    index.fit(vector_embeddings[chat_id])
    distances, indices = index.kneighbors(vec, n_neighbors=n_neighbors)
    return [vector_store[chat_id][i] for i in indices[0]]


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
                "Authorization": f"{os.getenv('RUNPOD_API_KEY')}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            async with session.post(RUNPOD_ENDPOINT + "/run", json=payload, headers=headers, timeout=30) as response:
                response.raise_for_status()
                data = await response.json()
                run_id = data.get("id")
                if not run_id:
                    logging.error("Не удалось получить ID задачи из ответа RunPod")
                    return "[Ой! Кажется, у меня техническая проблема под кодовым именем Арбузик]"

            status_url = f"{RUNPOD_ENDPOINT}/status/{run_id}"
            for _ in range(120):
                async with session.get(status_url) as status_resp:
                    status_resp.raise_for_status()
                    data = await status_resp.json()
                    if data.get("status") == "COMPLETED":
                        return data.get("output",
                                        "[Ой! Кажется, у меня техническая проблема под кодовым именем Клубничка]")
                    elif data.get("status") == "FAILED":
                        logging.error(f"Ошибка выполнения задачи: {data}")
                        return f"[Ой! Кажется, у меня техническая проблема под кодовым именем Абрикосик]"
                await asyncio.sleep(1)

            return "[Ой! Кажется, у меня техническая проблема под кодовым именем Сливка]"

    except Exception as e:
        logging.error(f"Ошибка при обращении к RunPod: {e}")
        return f"[Ой! Кажется, у меня техническая проблема под кодовым именем Персик]"


# === ОБРАБОТКА СООБЩЕНИЙ ===
@dp.message()
async def handle_message(message: Message):
    logging.info(f"Входящее сообщение от {message.chat.id}: {message.text}")
    chat_id = message.chat.id
    user_input = message.text.strip()

    if user_input == "/start":
        logging.info(f"Команда /start от {chat_id}")
        chat_history[chat_id] = []
        await message.answer("Привет, я Ника! А тебя как зовут?")
        return

    await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    if chat_id not in chat_history:
        chat_history[chat_id] = []
    if chat_id not in vector_store:
        vector_store[chat_id] = []
        vector_embeddings[chat_id] = []

    chat_history[chat_id].append(f"Ты: {user_input}")
    emb = embed_text(user_input)
    vector_store[chat_id].append(user_input)
    vector_embeddings[chat_id].append(emb)

    if len(chat_history[chat_id]) > MAX_HISTORY_SIZE:
        chat_history[chat_id] = chat_history[chat_id][-MAX_HISTORY_SIZE:]
        vector_store[chat_id] = vector_store[chat_id][-MAX_HISTORY_SIZE:]
        vector_embeddings[chat_id] = vector_embeddings[chat_id][-MAX_HISTORY_SIZE:]

    memories = find_similar(user_input, chat_id)
    history = truncate_history(chat_history[chat_id], CONTEXT_TOKEN_LIMIT)

    prompt = "\n".join(memories + history + ["Ника:"])
    reply = await run_llm(prompt)

    chat_history[chat_id].append(f"Ника: {reply}")
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
