import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# === LOAD ENV ===
# Загружаем переменные окружения из .env файла, если он существует
env_path = Path('.') / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# === КОНФИГУРАЦИЯ ===
# Считывание всех параметров из переменных окружения
BOT_NAME = os.getenv("BOT_NAME", "nika_ai_chatbot")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT", "")
CONTEXT_LENGTH = int(os.getenv("CONTEXT_LENGTH", 4096))
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 2500))
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
CHARACTER_CARD = os.getenv("CHARACTER_CARD", "")
CHARACTER_NAME = os.getenv("CHARACTER_NAME", "Арсен")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
                          "Ты виртуальный помощник, который помогает пользователям с их вопросами и задачами. Дружелюбный и отзывчивый.")
FIRST_MESSAGE = os.getenv("FIRST_MESSAGE", f"Привет, я {CHARACTER_NAME}.")
USER_FIRST_MESSAGE = os.getenv("USER_FIRST_MESSAGE",
                               "")  # Это сообщение, которое отправляется в историю сообщений от лица пользователя, чтобы не сломать шаблон инструкции LLM. Сам пользователь его не увидит, но в БД оно будет сохранено.
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MIN_P = float(os.getenv("MIN_P", 0.1))
TOP_P = float(os.getenv("TOP_P", 0.9))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", 1.1))
REPLY_MAX_TOKENS = int(os.getenv("REPLY_MAX_TOKENS", 400))
ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK = int(os.getenv("ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK", 10))
ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK = int(os.getenv("ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK", 40))
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "chat")
MAX_MESSAGES_PER_DAY = int(os.getenv("MAX_MESSAGES_PER_DAY", 50))
ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED = int(os.getenv("ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED", 100))
SUBSCRIBE_INVITE = os.getenv("SUBSCRIBE_INVITE", "")
EXCLUDE_WORDS = json.loads(os.getenv("EXCLUDE_WORDS", "[]"))
TOKENIZER_ENDPOINT = os.getenv("TOKENIZER_ENDPOINT", "")  # HTTP для токенизации текста
EMBEDDER_ENDPOINT = os.getenv("EMBEDDER_ENDPOINT", "")  # HTTP для векторизации текста
VECTOR_SIZE = 384  # Размерность для all-MiniLM-L6-v2
# Асинхронное подключение к MySQL
db_config = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "test_db")
}


def print_config():
    """Функция для вывода текущих настроек конфигурации."""
    logging.info(f"Настройки бота:\n"
                 f"RUNPOD_ENDPOINT: {RUNPOD_ENDPOINT}\n"
                 f"CONTEXT_LENGTH: {CONTEXT_LENGTH}\n"
                 f"CONTEXT_TOKEN_LIMIT: {CONTEXT_TOKEN_LIMIT}\n"
                 f"CHARACTER_CARD: {CHARACTER_CARD}\n"
                 f"SYSTEM_PROMPT: {SYSTEM_PROMPT}\n"
                 f"TEMPERATURE: {TEMPERATURE}\n"
                 f"MIN_P: {MIN_P}\n"
                 f"TOP_P: {TOP_P}\n"
                 f"REPEAT_PENALTY: {REPEAT_PENALTY}\n"
                 f"REPLY_MAX_TOKENS: {REPLY_MAX_TOKENS}\n"
                 f"RUNPOD_API_KEY: {RUNPOD_API_KEY}\n"
                 f"QDRANT_HOST: {QDRANT_HOST}\n"
                 f"QDRANT_PORT: {QDRANT_PORT}\n"
                 f"QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}\n"
                 f"VECTOR_SIZE: {VECTOR_SIZE}\n"
                 f"MYSQL_HOST: {db_config['host']}\n"
                 f"MYSQL_USER: {db_config['user']}\n"
                 f"MYSQL_DATABASE: {db_config['database']}\n"
                 f"MYSQL_PASSWORD: {'***' if db_config['password'] else 'не указано'}"
                 )
