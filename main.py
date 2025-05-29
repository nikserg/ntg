import asyncio
import logging
import os
import re
import uuid

import aiohttp
import aiomysql
from aiohttp import web
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import (db_config, print_config, QDRANT_HOST,
                    QDRANT_PORT, QDRANT_COLLECTION_NAME, VECTOR_SIZE,
                    SYSTEM_PROMPT, CHARACTER_CARD, CHARACTER_NAME,
                    ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK, ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED,
                    ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK,
                    MAX_MESSAGES_PER_DAY, TOKENIZER_ENDPOINT, EMBEDDER_ENDPOINT,
                    TEMPERATURE, MIN_P, TOP_P, REPEAT_PENALTY, REPLY_MAX_TOKENS,
                    RUNPOD_API_KEY, RUNPOD_ENDPOINT, SUBSCRIBE_INVITE, EXCLUDE_WORDS,
                    CONTEXT_TOKEN_LIMIT, USER_FIRST_MESSAGE, FIRST_MESSAGE,
                    BOT_NAME)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Хранилище для отслеживания сбора отзывов
collecting_feedback = {}

# Выводим параметры в лог
print_config()

# Инициализация клиента Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Комбинированный промпт для LLM
full_system_prompt = f"{SYSTEM_PROMPT}\n{CHARACTER_CARD}"

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

# Проверка и создание коллекции, если она не существует
def init_qdrant_collection():
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if QDRANT_COLLECTION_NAME not in collection_names:
        logging.info(f"Создание коллекции {QDRANT_COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        )


# Асинхронное подключение к MySQL с повторными попытками
async def get_db_connection(retries=3, delay=1):
    for attempt in range(1, retries + 1):
        try:
            return await aiomysql.connect(
                host=db_config["host"],
                user=db_config["user"],
                password=db_config["password"],
                db=db_config["database"],
                autocommit=True
            )
        except Exception as e:
            logging.error(f"Ошибка подключения к MySQL (попытка {attempt}): {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2  # экспоненциальная задержка
    return None

async def get_or_create_user(chat_id, invited_by=None):
    """Получает или создаёт пользователя"""
    query_check = "SELECT invite_code FROM users WHERE chat_id = %s"
    query_create = "INSERT INTO users (chat_id, invite_code, invited_by) VALUES (%s, %s, %s)"

    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query_check, (chat_id,))
            result = await cursor.fetchone()
            if result:
                return result[0]  # Возвращаем существующий инвайт-код

            # Создаём нового пользователя
            invite_code = str(uuid.uuid4())
            await cursor.execute(query_create, (chat_id, invite_code, invited_by))
            return invite_code


async def get_feedback_bonus(chat_id):
    """
    Возвращает кортеж (has_feedback, is_useful), где:
    - has_feedback: True, если есть хотя бы один фидбек
    - is_useful: True, если хотя бы один фидбек отмечен как полезный
    """
    query = "SELECT useful FROM feedbacks WHERE chat_id = %s LIMIT 1"
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            if result is None:
                return (False, False)
            is_useful = bool(result[0]) if result[0] is not None else False
            return (True, is_useful)

async def get_invited_users_count(chat_id):
    """Возвращает количество приглашённых пользователей"""
    query = "SELECT COUNT(*) FROM users WHERE invited_by = %s"
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            return result[0] if result else 0

async def get_user_daily_limit(chat_id):
    """Вычисляет дневной лимит сообщений для пользователя"""
    invited_count = await get_invited_users_count(chat_id)
    # Ограничиваем количество приглашений, учитываемых для бонуса
    bonus_count = min(invited_count, 5)
    # Проверяем наличие фидбека
    has_feedback, is_useful = await get_feedback_bonus(chat_id)
    feedback_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK if is_useful else ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK if has_feedback else 0
    return MAX_MESSAGES_PER_DAY + (ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED * bonus_count) + feedback_bonus

# Асинхронное получение количества сообщений за день
async def count_daily_messages(chat_id):
    query = """
            SELECT COUNT(*)
            FROM messages
            WHERE chat_id = %s
              AND DATE(time) = CURDATE() \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            return result[0] if result else 0


# Асинхронное сохранение сообщения
async def save_message(chat_id, message, role):
    query = """
            INSERT INTO messages (chat_id, message, role, is_current)
            VALUES (%s, %s, %s, %s) \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id, message, role, True))


# Асинхронное получение текущих сообщений
async def get_current_messages(chat_id):
    query = """
            SELECT message, role
            FROM messages
            WHERE chat_id = %s
              AND is_current = TRUE
            ORDER BY time ASC \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, (chat_id,))
            return await cursor.fetchall()


# Асинхронный сброс истории
async def reset_history(chat_id):
    query = """
            UPDATE messages
            SET is_current = FALSE
            WHERE chat_id = %s \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))

    # Удаление данных из Qdrant
    try:
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=str(chat_id)))]
            )
        )
        logging.info(f"Данные для chat_id={chat_id} успешно удалены из Qdrant.")
    except Exception as e:
        logging.error(f"Ошибка при удалении данных из Qdrant для chat_id={chat_id}: {e}")


def remove_newlines(text):
    """Удаляет переносы строк из текста, заменяя их на пробелы"""
    return text.replace("\n", " ")


def trim_incomplete_sentence(text):
    # Находит последнее завершённое предложение
    match = re.search(r'([.!?…\]*])[^.!?…\]*]*$', text)
    if match:
        end = match.end(1)
        return text[:end].strip()
    return text.strip()


def exclude_words_from_input(text, exclude_words):
    for word in exclude_words:
        # Удаляет слово только если оно встречается как отдельное слово (учитывает границы)
        text = re.sub(rf'\b{re.escape(word)}\b', '', text, flags=re.IGNORECASE)
    # Убираем лишние пробелы после удаления слов
    return re.sub(r'\s{2,}', ' ', text).strip()

def clean_llm_response(text):
    """Удаляет нежелательные символы или строки из ответа."""
    text = text.replace("***", "").strip()
    # Удаляем строку, начинающуюся с имени персонажа
    if text.startswith(f"{CHARACTER_NAME}:"):
        text = text[len(f"{CHARACTER_NAME}:"):].strip()
    # Обрезаем, если ответ заканчивается на '\n\nИмя:'
    text = re.split(r'\n\w+:', text)[0]
    # Обрезаем, если в конце есть объясниение
    text = re.split(r'\*+ОБЪЯСНЕНИЕ:', text)[0]
    # Обрезаем, если в сообщении есть упоминание развития сценария
    text = re.split(r'\(Сценарий ', text)[0]
    # Заменяем двойные пробелы на одинарные
    text = re.sub(r'\s{2,}', ' ', text)
    # Заменяем последовательность &amp; на амперсанд
    text = text.replace("&amp;", "&")
    # Заменяем последовательность &#34; на кавычки
    text = text.replace("&#34;", '"')
    # Заменяем HTML-сущности на соответствующие символы
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    # Убираем теги HTML
    text = re.sub("<s>", '', text)
    text = re.sub("</s>", '', text)
    # Убираем квадратные скобки
    text = re.sub(r"\[", '', text)
    text = re.sub("]", '', text)
    # Убираем угловые скобки
    text = re.sub("<", '', text)
    text = re.sub(">", '', text)
    return text.strip()


async def truncate_history(messages, max_tokens):
    """
    Обрезает историю сообщений, чтобы оставить максимальное
    количество последних сообщений в рамках max_tokens.
    """
    if not messages:
        return []

    # Подсчёт токенов для каждого сообщения
    total_tokens = 0
    truncated = []

    for msg in reversed(messages):
        # Обращаемся к отдельному сервису для токенизации, повторяем попытки при ошибках
        async with aiohttp.ClientSession() as session:
            retries = 3
            for attempt in range(retries):
                try:
                    async with session.post(TOKENIZER_ENDPOINT, json={"text": msg["message"]}, timeout=30) as response:
                        response.raise_for_status()
                        data = await response.json()
                        msg_tokens = data.get("tokens", 0)
                        break
                except Exception as e:
                    logging.error(f"Ошибка при токенизации сообщения: {e}")
                    if attempt == retries - 1:
                        msg_tokens = len(msg["message"])  # Если не удалось получить токены, используем длину сообщения
                    await asyncio.sleep(2 * (attempt + 1))  # экспоненциальная задержка

        if total_tokens + msg_tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total_tokens += msg_tokens

    return truncated


async def embed_text(text):
    # Обращаемся к отдельному сервису для векторизации, повторяем попытки при ошибках
    async with aiohttp.ClientSession() as session:
        retries = 3
        for attempt in range(retries):
            try:
                async with session.post(EMBEDDER_ENDPOINT, json={"text": text}, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
                    embeddings = data.get("embeddings", [])
                    return embeddings
            except Exception as e:
                logging.error(f"Ошибка при векторизации текста: {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
                return None
        return None

async def find_similar(text, chat_id, current_context=None, top_k=3):
    # Генерируем эмбеддинг для поиска
    query_vector = await embed_text(text)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        find_similar_sync,
        chat_id, query_vector, current_context, top_k
    )


def find_similar_sync(chat_id, query_vector, current_context=None, top_k=3):
    """Находит похожие сообщения в Qdrant, исключая те, что уже есть в контексте"""
    if current_context is None:
        current_context = []

    try:
        # Ищем похожие сообщения в Qdrant только для данного chat_id
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=str(chat_id)))]
            ),
            limit=top_k * 3  # Запрашиваем больше результатов для фильтрации
        )

        # Фильтруем результаты, исключая те, что уже в контексте
        similar_msgs = []
        for result in search_result:
            candidate = result.payload["text"]
            role_name = "Собеседник" if result.payload["role"] == "user" else CHARACTER_NAME
            if candidate not in current_context:
                similar_msgs.append(f"{role_name}: {candidate}")
                if len(similar_msgs) >= top_k:
                    break

        return similar_msgs
    except Exception as e:
        logging.error(f"Ошибка при поиске в Qdrant: {e}")
        return []


async def save_message_to_qdrant(chat_id, message_text, role):
    """Сохраняет сообщение в Qdrant"""
    try:
        # Генерируем уникальный ID для сообщения (UUID)
        message_id = str(uuid.uuid4())
        emb = await embed_text(message_text)
        # Сохраняем сообщение и его вектор
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=message_id,
                    vector=emb,
                    payload={"chat_id": str(chat_id), "text": message_text, "role": role}
                )
            ]
        )
        return True
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Qdrant: {e}")
        return False


async def run_llm(messages):
    payload = {
        "input": {
            "messages": messages,
            "params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "min_p": MIN_P,
                "repeat_penalty": REPEAT_PENALTY,
                "max_tokens": REPLY_MAX_TOKENS
            }
        }
    }
    logging.info(f"Отправка запроса к LLM: {payload}")
    retries = 3  # Количество попыток при ошибках
    delay = 2  # Начальная задержка между попытками
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                async with session.post(RUNPOD_ENDPOINT + "/runsync", json=payload, headers=headers,
                                        timeout=240) as response:
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
                            async with session.get(f"{RUNPOD_ENDPOINT}/status/{task_id}",
                                                   headers=headers, timeout=240) as status_response:
                                status_data = await status_response.json()
                                status = status_data.get("status")

                                if status == "COMPLETED":
                                    output = status_data.get("output", {})
                                    text = output.get("text")
                                    if text is None:
                                        logging.error(f"Ответ LLM без текста: {status_data}")
                                        return "[Ой! Кажется, у меня техническая проблема под кодовым именем Клубничка]"
                                    return clean_llm_response(text)

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
                    return clean_llm_response(text)
        except Exception as e:
            logging.error(f"Ошибка при обращении к RunPod (попытка {attempt}): {e}")
            if attempt == retries:
                return "[Ой! Кажется, у меня техническая проблема под кодовым именем Персик]"
            await asyncio.sleep(delay)
            delay *= 2  # экспоненциальная задержка
    return None


async def build_messages(chat_id, user_input):
    """
    Формирует список сообщений для LLM из истории чата, векторной БД и системного промпта.
    """
    # Получаем историю чата из MySQL
    history_records = await get_current_messages(chat_id)

    # Обрезаем историю до лимита токенов
    history = await truncate_history(history_records, CONTEXT_TOKEN_LIMIT)

    # Находим похожие сообщения из векторной БД
    memories = await find_similar(user_input, chat_id, current_context=[msg["message"] for msg in history])

    # Формируем системное сообщение
    system_message = {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n***\n{CHARACTER_CARD}" + (
            f"\n***\nПредыдущие сообщения:\n" + "\n".join(memories) if memories else "")
    }

    # Формируем сообщения из истории чата
    messages = [system_message]

    for msg in history:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["message"]})
        else:
            messages.append({"role": "user", "content": msg["message"]})

    return messages


async def collect_feedback(chat_id, text):
    feedback_text = text.strip()
    if not feedback_text:
        return "Отзыв не может быть пустым. Попробуйте снова.", False
    # Проверяем, не превышает ли отзыв максимальную длину
    if len(feedback_text) > 2000:
        return "Отзыв слишком длинный. Пожалуйста, сократите его до 2000 символов.", False
    # Проверяем минимальную длину отзыва
    if len(feedback_text) < 10:
        return "Отзыв слишком короткий. Пожалуйста, напишите что-нибудь более развернутое.", False
    # Проверяем фидбек на мусорность: если в нем слишком много повторяющихся символов (типа "Это печка!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" или "аааааааааааааааааааа")
    if re.search(r'(.)\1{9,}', feedback_text):
        return "Похоже, в вашем отзыве слишком много повторяющихся символов. Пожалуйста, напишите что-то более осмысленное.", False

    query = """
                INSERT INTO feedbacks (chat_id, feedback)
                VALUES (%s, %s)
                """
    try:
        async with (await get_db_connection()) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(query, (chat_id, feedback_text))
        return f"Спасибо за ваш отзыв! Мы ценим ваше мнение. Вы можете оставить новый отзыв в любое время с помощью команды /feedback. А сейчас вы можете продолжить чат.\n\n{SUBSCRIBE_INVITE}", True
    except Exception as e:
        logging.error(f"Ошибка при записи отзыва в БД: {e}")
        return "Не удалось сохранить отзыв. Попробуйте позже.", False

async def handle_internal_request(request):
    """Обработчик для внутреннего API эндпойнта."""
    try:
        # Получаем текст запроса напрямую из тела
        data = await request.json()
        user_input = data["text"].strip()
        chat_id = data["chat_id"]
        logging.info(f"Входящее сообщение от {chat_id}: {user_input}")

        # Проверяем, собираем ли отзыв
        if collecting_feedback.get(chat_id, False):
            logging.info(f"Сбор отзыва от {chat_id}: {user_input}")
            response, successful_feedback = await collect_feedback(chat_id, user_input)
            if successful_feedback:
                collecting_feedback[chat_id] = False  # Сбрасываем статус сбора отзыва
            return web.Response(text=response)

        if user_input.startswith("/start"):
            logging.info(f"Команда {user_input} от {chat_id}")
            # Проверяем наличие инвайт-кода
            parts = user_input.split()
            invited_by = None

            if len(parts) > 1:
                invite_code = parts[1]
                # Ищем пользователя с таким инвайт-кодом
                async with (await get_db_connection()) as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("SELECT chat_id FROM users WHERE invite_code = %s", (invite_code,))
                        result = await cursor.fetchone()
                        if result and result[0] != chat_id:
                            invited_by = result[0]

            # Регистрируем или получаем пользователя
            invite_code = await get_or_create_user(chat_id, invited_by)

            # Сбрасываем историю сообщений и отправляем приветственное сообщение
            await reset_history(chat_id)
            first_message = FIRST_MESSAGE.replace('\\n', '\n')
            await save_message(chat_id, USER_FIRST_MESSAGE,
                               "user")  # Это сообщение не будет видно пользователю, но сохранится в БД
            await save_message(chat_id, remove_newlines(first_message), "assistant")
            return web.Response(text=FIRST_MESSAGE.replace("\\n", "\n"))

        if user_input == "/feedback":
            collecting_feedback[chat_id] = True  # Устанавливаем статус сбора отзыва
            return web.Response(
                text=f"Бот сейчас в бета-версии, поэтому мы особенно ценим отзывы. За отзыв вы можете получить от {ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK} до {ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK} сообщений в день. Пожалуйста, напишите, что вам понравилось, что можно улучшить или какие ошибки вы заметили.")

        if user_input == "/invite":
            # Получаем или создаем пользователя для получения invite_code
            invite_code = await get_or_create_user(chat_id)
            invite_link = f"https://t.me/{BOT_NAME}?start={invite_code}"

            return web.Response(text=
                                f"Поделитесь этой ссылкой с друзьями и получите дополнительно "
                                f"{ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED} сообщений в день "
                                f"за каждого приглашённого пользователя:\n\n{invite_link}"
                                f"\n\n{SUBSCRIBE_INVITE}"
                                )

        if user_input == "/limit":
            # Получаем текущие показатели
            daily_message_count = await count_daily_messages(chat_id)
            invited_count = await get_invited_users_count(chat_id)
            bonus_count = min(invited_count, 5)
            daily_limit = await get_user_daily_limit(chat_id)

            # Получаем бонусы за фидбек
            has_feedback, is_useful = await get_feedback_bonus(chat_id)
            feedback_bonus_str = ""
            if has_feedback:
                feedback_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK if is_useful else ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK
                feedback_type = "полезный" if is_useful else "обычный"
                feedback_bonus_str = f"• Бонус за отзыв: +{feedback_bonus} ({feedback_type})\n"

            # Вычисляем базовый лимит и бонус от приглашений
            base_limit = MAX_MESSAGES_PER_DAY
            invite_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED * bonus_count

            return web.Response(text=
                                f"Ваш лимит сообщений на сегодня: {daily_limit}\n"
                                f"• Базовый лимит: {base_limit}\n"
                                f"• Бонус за приглашения: +{invite_bonus} ({invited_count} приглашённых, учитывается максимум 5)\n"
                                f"{feedback_bonus_str}"
                                f"Использовано сегодня: {daily_message_count} из {daily_limit}\n"
                                f"\n"
                                f"Используйте /invite для получения ссылки-приглашения и увеличения лимита.\n"
                                f"Оставьте отзыв с помощью /feedback, чтобы получить дополнительный бонус!"
                                f"\n\n{SUBSCRIBE_INVITE}"
                                )

        # Проверяем, не превышен ли лимит сообщений
        daily_message_count = await count_daily_messages(chat_id)
        user_daily_limit = await get_user_daily_limit(chat_id)

        if daily_message_count >= user_daily_limit:
            # Получаем инвайт-код пользователя
            invite_code = await get_or_create_user(chat_id)
            invite_link = f"https://t.me/{BOT_NAME}?start={invite_code}"

            return web.Response(text=
                                f"Вы достигли лимита бета-версии в {user_daily_limit} сообщений за сегодня. "
                                f"Счетчик будет сброшен завтра.\n\n"
                                f"Поделитесь своим опытом с помощью команды /feedback, чтобы получить от {ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK} до {ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK} дополнительных сообщений в день!\n\n"
                                f"А также вы можете пригласить друзей по этой ссылке и получить дополнительно "
                                f"{ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED} сообщений в день за каждого приглашённого:\n{invite_link}\n\n"
                                f"{SUBSCRIBE_INVITE}")

        if not await get_current_messages(chat_id):
            await save_message(chat_id, USER_FIRST_MESSAGE,
                               "user")  # Это сообщение не будет видно пользователю, но сохранится в БД
            first_message = FIRST_MESSAGE.replace('\\n', '\n')
            await save_message(chat_id, remove_newlines(first_message), "assistant")

        cleaned_input = remove_newlines(user_input)
        cleaned_input = exclude_words_from_input(cleaned_input, EXCLUDE_WORDS)
        await save_message(chat_id, cleaned_input, "user")
        # Добавляем сообщение пользователя в Qdrant
        await save_message_to_qdrant(chat_id, cleaned_input, "user")

        # Формируем сообщения для LLM
        messages = await build_messages(chat_id, cleaned_input)

        # Отправляем запрос к LLM
        reply = await run_llm(messages)
        reply = trim_incomplete_sentence(reply)
        cleaned_reply = remove_newlines(reply)
        # Добавляем ответ бота в векторное хранилище
        await save_message_to_qdrant(chat_id, cleaned_reply, "assistant")
        await save_message(chat_id, reply, "assistant")
        logging.info(f"Ответ пользователю {chat_id}: {reply}")
        return web.Response(text=reply)
    except Exception as e:
        logging.error(f"Ошибка при обработке /internal: {e}")
        return web.Response(text=f"Ошибка: {e}", status=500)


async def apply_migrations():
    """Применяет миграции для базы данных."""
    create_messages_table = """
        CREATE TABLE IF NOT EXISTS messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            message TEXT NOT NULL,
            role VARCHAR(50) NOT NULL,
            is_current BOOLEAN DEFAULT TRUE,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    create_feedbacks_table = """
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            feedback TEXT NOT NULL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """
    create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            chat_id BIGINT PRIMARY KEY,
            invite_code VARCHAR(36) UNIQUE NOT NULL,
            invited_by BIGINT DEFAULT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (invited_by)
        );
    """
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(create_messages_table)
            await cursor.execute(create_feedbacks_table)
            await cursor.execute(create_users_table)

    # Добавляем индексы отдельно, чтобы не вызывать ошибки при повторном запуске
    add_indexes = [
        "ALTER TABLE messages ADD INDEX idx_messages_chat_current (chat_id, is_current);",
        "ALTER TABLE messages ADD INDEX idx_messages_chat_time (chat_id, time);",
        "ALTER TABLE feedbacks ADD INDEX idx_feedbacks_chat_id (chat_id);"
    ]
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            for index_query in add_indexes:
                try:
                    await cursor.execute(index_query)
                except Exception as e:
                    # Игнорируем ошибки, если индекс уже существует
                    logging.info(f"При создании индекса: {e}")
    # Добавляем столбец за полезный фидбек, если его нет
    alter_feedbacks_table = """
        ALTER TABLE feedbacks ADD COLUMN useful BOOLEAN DEFAULT FALSE;
    """
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.execute(alter_feedbacks_table)
            except Exception as e:
                # Игнорируем ошибки, если столбец уже существует
                logging.info(f"При добавлении столбца useful в таблицу feedbacks: {e}")

# === ЗАПУСК WEBHOOK ===
async def on_startup(app):
    await asyncio.get_event_loop().run_in_executor(None, init_qdrant_collection)
    # Применяем миграции
    await apply_migrations()

# Создание и запуск aiohttp-приложения
app = web.Application()
app.on_startup.append(on_startup)
app.router.add_post('/internal', handle_internal_request)

if __name__ == "__main__":
    web.run_app(app, host="::", port=int(os.getenv("PORT", 8080)))
