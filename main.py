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
import time

# === LOAD ENV ===
# Загружаем переменные окружения из .env файла, если он существует
from pathlib import Path
from dotenv import load_dotenv

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
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", 8192))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 1536))
MAX_HISTORY_SIZE = int(os.getenv("MAX_HISTORY_SIZE", 1000))
CHARACTER_CARD = os.getenv("CHARACTER_CARD", "Name: Ника\nPersonality: Игривая, кокетливая, милая, немного дерзкая\nAppearance: Розовые волосы, пронзительные зелёные глаза, кружевной чокер\nRole: Ролевая собеседница, отвечает от первого лица")
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", "/webhook")
WEBHOOK_BASE = os.getenv("WEBHOOK_BASE")
WEBHOOK_URL = WEBHOOK_BASE + WEBHOOK_PATH if WEBHOOK_BASE else None
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — персонаж по имени Ника. Говоришь от первого лица. Отвечай естественно, как будто ты реальный человек. Ролевая беседа.")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MIN_P = float(os.getenv("MIN_P", 0.1))

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
    # Ищет похожие сообщения по векторной близости
    if chat_id not in vector_embeddings or not vector_embeddings[chat_id]:
        return []
    vec = embed_text(text).reshape(1, -1)
    index = NearestNeighbors(n_neighbors=top_k, algorithm='auto', metric='euclidean')
    index.fit(vector_embeddings[chat_id])
    distances, indices = index.kneighbors(vec, n_neighbors=min(top_k, len(vector_embeddings[chat_id])))
    return [vector_store[chat_id][i] for i in indices[0]]

def run_llm(prompt):
    # Асинхронная обработка запроса к RunPod через run/status
    payload = {
        "input": {
            "prompt": f"{full_system_prompt}\n{prompt}",
            "temperature": TEMPERATURE,
            "top_p": MIN_P
        }
    }
    logging.info(f"Отправка запроса к LLM: {payload}")
    try:
        response = requests.post(RUNPOD_ENDPOINT + "/run", json=payload)
        response.raise_for_status()
        run_id = response.json().get("id")
        if not run_id:
            return "[Ошибка: не получен идентификатор задачи RunPod]"

        # Ожидание завершения задачи
        status_url = f"{RUNPOD_ENDPOINT}/status/{run_id}"
        for _ in range(120):
            status_resp = requests.get(status_url)
            status_resp.raise_for_status()
            data = status_resp.json()
            if data.get("status") == "COMPLETED":
                return data.get("output", "[Ошибка: пустой вывод]")
            elif data.get("status") == "FAILED":
                return f"[Ошибка: задача завершилась с ошибкой: {data}]"
            time.sleep(1)

        return "[Ошибка: превышено время ожидания ответа от RunPod]"

    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при обращении к RunPod: {e}")
        return f"[Ошибка запроса: {e}]"
    except ValueError:
        logging.error("Ошибка разбора JSON-ответа от RunPod")
        return "[Ошибка: RunPod не вернул корректный JSON]"

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
    reply = run_llm(prompt)

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
