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
# CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 2500))  # Ставим чуть меньше, чем 4096, чтобы учесть неточность подсчета токенов
CONTEXT_TOKEN_LIMIT = int(os.getenv("CONTEXT_TOKEN_LIMIT", 1000))  # Для теста пересказа
SUMMARIZE_BUFFER_PERCENT = int(os.getenv("SUMMARIZE_BUFFER_PERCENT", 20))
SUMMARIZE_TARGET_TOKEN_LENGTH = int(os.getenv("SUMMARIZE_TARGET_TOKEN_LENGTH", 400))  # Количество токенов для пересказа
SUMMARIZE_TEMPERATURE = float(os.getenv("SUMMARIZE_TEMPERATURE", 0.4))  # Температура для пересказа
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
                          "Ты виртуальный помощник, который помогает пользователям с их вопросами и задачами. Дружелюбный и отзывчивый.")
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
ADDITIONAL_MESSAGES_PER_DAY_FOR_SUBSCRIPTION = int(os.getenv("ADDITIONAL_MESSAGES_PER_DAY_FOR_SUBSCRIPTION", 15))
SUBSCRIBE_INVITE = os.getenv("SUBSCRIBE_INVITE", "")
EXCLUDE_WORDS = json.loads(os.getenv("EXCLUDE_WORDS", "[]"))
TOKENIZER_ENDPOINT = os.getenv("TOKENIZER_ENDPOINT", "")  # HTTP для токенизации текста
EMBEDDER_ENDPOINT = os.getenv("EMBEDDER_ENDPOINT", "")  # HTTP для векторизации текста
SUBSCRIBE_CHECK_ENDPOINT = os.getenv("SUBSCRIBE_CHECK_ENDPOINT", "")
VECTOR_SIZE = 384  # Размерность для all-MiniLM-L6-v2
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "test_db")
}


def print_config():
    """Функция для вывода текущих настроек конфигурации."""
    logging.info(f"Настройки бота:\n"
                 f"RUNPOD_ENDPOINT: {RUNPOD_ENDPOINT}\n"
                 f"CONTEXT_TOKEN_LIMIT: {CONTEXT_TOKEN_LIMIT}\n"
                 f"SYSTEM_PROMPT: {SYSTEM_PROMPT}\n"
                 f"TEMPERATURE: {TEMPERATURE}\n"
                 f"MIN_P: {MIN_P}\n"
                 f"TOP_P: {TOP_P}\n"
                 f"REPEAT_PENALTY: {REPEAT_PENALTY}\n"
                 f"REPLY_MAX_TOKENS: {REPLY_MAX_TOKENS}\n"
                 f"RUNPOD_API_KEY: {RUNPOD_API_KEY}\n"
                 f"QDRANT_COLLECTION_NAME: {QDRANT_COLLECTION_NAME}\n"
                 f"VECTOR_SIZE: {VECTOR_SIZE}\n"
                 )
