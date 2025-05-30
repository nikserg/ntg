import asyncio
import logging
import uuid

import aiomysql

from config import db_config


async def get_db_connection(retries=3, delay=1):
    """Асинхронное подключение к базе данных MySQL с поддержкой повторных попыток"""
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


async def execute_query(query, params=None):
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, params)


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


# Асинхронное сохранение сообщения
async def save_message(chat_id, message, role):
    """Сохраняет сообщение в базе данных"""
    # Получаем текущий диалог для чата
    dialogue_id = await get_current_dialogue(chat_id)
    query = """
            INSERT INTO messages (dialogue_id, message, role, chat_id)
            VALUES (%s, %s, %s, %s) \
            """
    await execute_query(query, (dialogue_id, message, role, chat_id))


async def get_current_dialogue(chat_id):
    """Получает текущий диалог для чата"""
    query = """
            SELECT id FROM dialogues
            WHERE chat_id = %s AND is_current = TRUE \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            if result:
                return result["id"]
    # Если диалог не найден, можно создать новый
    query = """
            INSERT INTO dialogues (chat_id)
            VALUES (%s) \
            """
    await execute_query(query, (chat_id,))
    return await get_current_dialogue(chat_id)


async def get_current_messages(chat_id):
    """Получает текущие сообщения для чата"""
    query = """
            SELECT messages.message, messages.role
            FROM messages
            LEFT JOIN dialogues ON messages.dialogue_id = dialogues.id
            WHERE dialogues.chat_id = %s
              AND dialogues.is_current = TRUE
            ORDER BY messages.time ASC \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, (chat_id,))
            return await cursor.fetchall()
