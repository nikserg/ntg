import logging

import aiomysql

from characters import get_character
from config import SUMMARIZE_BUFFER_PERCENT
from db import execute_query, get_db_connection
from dialogues import get_current_dialogue


def get_summarize_buffer(messages):
    """
    Рассчитывает буфер для пересказа сообщений.
    Используется для определения, сколько сообщений нужно пересказать,
    чтобы не превысить лимит токенов.
    """
    messages_count_to_summarize = int(len(messages) * (SUMMARIZE_BUFFER_PERCENT / 100))
    # Получаем самые старые сообщения для пересказа
    messages_to_summarize = messages[:messages_count_to_summarize]
    return messages_to_summarize


async def mark_messages_as_summarized(messages):
    """
    Помечает сообщения как пересказанные в базе данных.
    """
    message_ids = [msg["id"] for msg in messages]
    placeholders = ', '.join(['%s'] * len(message_ids))
    query = f"UPDATE messages SET summarized = 1 WHERE id IN ({placeholders})"
    try:
        await execute_query(query, message_ids)
    except Exception as e:
        logging.error("Ошибка при пометке сообщений как пересказанных: %s", e)
        raise e


async def write_summary_to_db(chat_id, summary):
    """
    Записывает пересказ в базу данных.
    """
    dialogue_id = await get_current_dialogue(chat_id)
    query = "INSERT INTO summaries (dialogue_id, summary) VALUES (%s, %s)"
    try:
        await execute_query(query, (dialogue_id, summary))
    except Exception as e:
        logging.error("Ошибка при записи пересказа в базу данных: %s", e)
        raise e


async def get_summary(chat_id):
    """
    Получает последний пересказ для чата.
    """
    dialogue_id = await get_current_dialogue(chat_id)
    query = "SELECT summary FROM summaries WHERE dialogue_id = %s ORDER BY id DESC LIMIT 1"
    async with (await get_db_connection()) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, (dialogue_id,))
            result = await cursor.fetchone()
            if result:
                return result["summary"]
    # Если пересказ не найден, возвращаем первый пересказ для персонажа
    character = await get_character(chat_id)
    summary = character.get("first_summary", "")
    # Сохраним первый пересказ в базе данных, раз он не существует
    try:
        await write_summary_to_db(chat_id, summary)
    except Exception as e:
        logging.error("Ошибка при сохранении первого пересказа в базу данных: %s", e)
        raise e
    return summary
