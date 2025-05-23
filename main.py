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
import os

# === LOAD ENV ===
# Загружаем переменные окружения из .env файла (если он существует)
from pathlib import Path
from dotenv import load_dotenv

env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Настройка логирования — выводит информацию в консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# === CONFIG ===
# Получаем конфигурационные значения из переменных окружения
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 8192))
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
WEBHOOK_URL = WEBHOOK_BASE + WEBHOOK_PATH if WEBHOOK_BASE else None
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ника — дружелюбный и немного игривый ИИ-собеседник.")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MIN_P = float(os.getenv("MIN_P", 0.1))

# Логируем конфигурацию
logging.info(f"Telegram Token: {TELEGRAM_TOKEN}")
logging.info(f"Runpod Endpoint: {RUNPOD_ENDPOINT}")
logging.info(f"Model Max Tokens: {MODEL_MAX_TOKENS}")
logging.info(f"Webhook URL: {WEBHOOK_URL}")
logging.info(f"Webhook Base: {WEBHOOK_BASE}")
logging.info(f"System Prompt: {SYSTEM_PROMPT}")
logging.info(f"Temperature: {TEMPERATURE}")
logging.info(f"Min P: {MIN_P}")

# === SETUP ===
# Инициализация Telegram-бота, хранилища и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Подгружаем токенизатор и модель SentenceTransformer для эмбеддингов
tokenizer = tiktoken.get_encoding("cl100k_base")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Словари для хранения истории чата и эмбеддингов
vector_store = {}
vector_embeddings = {}
chat_history = {}

# === HELPERS ===
def truncate_history(messages, max_tokens):
    # Обрезает историю сообщений, чтобы вписаться в лимит токенов
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
    # Создаёт эмбеддинг из текста
    return embedder.encode([text])[0].astype("float32")

def find_similar(text, chat_id, top_k=3):
    # Находит похожие сообщения из прошлого по векторному сходству
    if chat_id not in vector_embeddings or not vector_embeddings[chat_id]:
        return []
    vec = embed_text(text).reshape(1, -1)
    index = NearestNeighbors(n_neighbors=top_k, algorithm='auto', metric='euclidean')
    index.fit(vector_embeddings[chat_id])
    distances, indices = index.kneighbors(vec, n_neighbors=min(top_k, len(vector_embeddings[chat_id])))
    return [vector_store[chat_id][i] for i in indices[0]]

def run_llm(prompt):
    # Отправляет промпт на Runpod и возвращает ответ модели
    payload = {
        "input": {
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "temperature": TEMPERATURE,
            "top_p": MIN_P
        }
    }
    logging.info(f"Отправка запроса к LLM: {payload}")
    response = requests.post(RUNPOD_ENDPOINT, json=payload)
    try:
        return response.json()["output"]
    except Exception:
        return "[Ошибка: модель не вернула корректный ответ]"

# === HANDLERS ===
@dp.message()
async def handle_message(message: Message):
    # Обрабатывает все входящие сообщения Telegram
    logging.info(f"Входящее сообщение от {message.chat.id}: {message.text}")
    chat_id = message.chat.id
    user_input = message.text.strip()

    # Обработка команды /start
    if user_input == "/start":
        logging.info(f"Команда /start от {chat_id}")
        chat_history[chat_id] = []
        await message.answer("Привет, я Ника! А тебя как зовут?")
        return

    # Инициализация истории и эмбеддингов, если пользователь новый
    if chat_id not in chat_history:
        chat_history[chat_id] = []
    if chat_id not in vector_store:
        vector_store[chat_id] = []
        vector_embeddings[chat_id] = []

    # Добавляем сообщение пользователя в историю и векторное хранилище
    chat_history[chat_id].append(f"Ты: {user_input}")
    emb = embed_text(user_input)
    vector_store[chat_id].append(user_input)
    vector_embeddings[chat_id].append(emb)

    # Ищем релевантные воспоминания и формируем историю
    memories = find_similar(user_input, chat_id)
    history = truncate_history(chat_history[chat_id], MODEL_MAX_TOKENS - 256)

    # Генерация ответа от модели
    prompt = "\n".join(memories + history + ["Ника:"])
    reply = run_llm(prompt)

    # Сохраняем ответ и отправляем пользователю
    chat_history[chat_id].append(f"Ника: {reply}")
    logging.info(f"Ответ пользователю {chat_id}: {reply}")
    await message.answer(reply)

# === WEBHOOK SETUP ===
async def on_startup(app):
    # Устанавливаем webhook при запуске сервера
    if WEBHOOK_URL:
        logging.info(f"Установка вебхука на {WEBHOOK_URL}")
        await bot.set_webhook(WEBHOOK_URL)
    else:
        logging.warning("WEBHOOK_URL не указан. Вебхук не будет установлен.")

async def on_shutdown(app):
    # Удаляем webhook при завершении работы
    if WEBHOOK_URL:
        await bot.delete_webhook()

# Настройка aiohttp-приложения и регистрация webhook
app = web.Application()
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if WEBHOOK_URL:
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)

# Запуск aiohttp-приложения
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
