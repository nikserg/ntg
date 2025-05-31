import aiomysql

from characters import get_character
from config import SUMMARIZE_BUFFER_PERCENT
from db import execute_query, get_db_connection
from dialogues import get_current_dialogue


def summary_system_prompt(previous_summary):
    return f"Перескажи текущие сообщения в диалоге, сохранив суть и ключевые моменты. Предыдущий пересказ: {previous_summary}"


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
    query = "UPDATE messages SET summarized = 1 WHERE id IN %s"
    message_ids = tuple(msg["id"] for msg in messages)
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (message_ids))
            await conn.commit()


async def write_summary_to_db(chat_id, summary):
    """
    Записывает пересказ в базу данных.
    """
    dialogue_id = await get_current_dialogue(chat_id)
    query = "INSERT INTO summaries (dialogue_id, summary) VALUES (%s, %s)"
    await execute_query(query, (dialogue_id, summary))


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
    await write_summary_to_db(chat_id, summary)
    return summary
