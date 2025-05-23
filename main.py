import logging
import requests
import tiktoken
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os

# === LOAD ENV ===
# Загружаем переменные окружения из .env, если файл существует
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# === CONFIG ===
# Читаем конфигурационные переменные из окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 2048))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
WEBHOOK_URL = WEBHOOK_BASE + WEBHOOK_PATH if WEBHOOK_BASE else None

# === SETUP ===
# Настройка логгирования и компонентов бота
logging.basicConfig(level=logging.INFO)
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Загружаем токенизатор и эмбеддер
tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Настройка поиска ближайших векторов
index = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='euclidean')
vector_store = []  # Список сохранённых сообщений
vector_embeddings = []  # Эмбеддинги сообщений
chat_history = {}  # История чатов

# === HELPERS ===
def truncate_history(messages, max_tokens):
    """Обрезает историю сообщений так, чтобы она помещалась в лимит токенов"""
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
    """Создаёт эмбеддинг из текста"""
    return embedder.encode([text])[0].astype("float32")

def find_similar(text, top_k=3):
    """Находит наиболее похожие сообщения по смыслу"""
    if not vector_embeddings:
        return []
    vec = embed_text(text).reshape(1, -1)
    index.fit(vector_embeddings)
    distances, indices = index.kneighbors(vec, n_neighbors=min(top_k, len(vector_embeddings)))
    return [vector_store[i] for i in indices[0]]

def run_llm(prompt):
    """Отправляет запрос к LLM и получает ответ"""
    response = requests.post(RUNPOD_ENDPOINT, json={"input": {"prompt": prompt}})
    try:
        return response.json()["output"]
    except Exception:
        return "[Ошибка: LLM не ответила корректно]"

# === HANDLERS ===
@dp.message()
async def cmd_start(message: Message):
    """Обработка команды /start — начинаем новую сессию"""
    chat_history[message.chat.id] = []
    await message.answer("Привет! Пиши что-нибудь, и я отвечу в стиле ролплея ✨")

@dp.message()
async def handle_message(message: Message):
    logging.info(f"Входящее сообщение от {message.chat.id}: {message.text}")
    """Основной обработчик сообщений"""
    chat_id = message.chat.id
    user_input = message.text.strip()

    if chat_id not in chat_history:
        chat_history[chat_id] = []

    # Добавляем сообщение пользователя в историю
    chat_history[chat_id].append(f"Ты: {user_input}")

    # Обновляем векторную базу
    emb = embed_text(user_input)
    vector_store.append(user_input)
    vector_embeddings.append(emb)

    # Ищем релевантные воспоминания
    memories = find_similar(user_input)

    # Обрезаем историю под лимит токенов
    history = truncate_history(chat_history[chat_id], MODEL_MAX_TOKENS - 256)

    # Составляем prompt для модели
    prompt = "\n".join(memories + history + ["Бот:"])

    # Запрашиваем ответ у модели
    reply = run_llm(prompt)

    # Сохраняем и отправляем ответ
    chat_history[chat_id].append(f"Бот: {reply}")
    logging.info(f"Ответ пользователю {chat_id}: {reply}")
    await message.answer(reply)

# === WEBHOOK SETUP ===
async def on_startup(app):
    """Устанавливаем webhook при запуске приложения, если указан"""
    if WEBHOOK_URL:
        await bot.set_webhook(WEBHOOK_URL)
    else:
        logging.warning("WEBHOOK_URL не указан. Вебхук не будет установлен.")

async def on_shutdown(app):
    """Удаляем webhook при завершении работы, если он был установлен"""
    if WEBHOOK_URL:
        await bot.delete_webhook()

# Создаём и запускаем aiohttp-приложение
app = web.Application()
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# Подключаем бота к webhook через SimpleRequestHandler, если задан
if WEBHOOK_URL:
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)

# Запуск веб-сервера
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
