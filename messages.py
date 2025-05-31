from db import *
from dialogues import get_current_dialogue
from tokenizer import count_tokens


# Асинхронное сохранение сообщения
async def save_message(chat_id, message, role):
    """Сохраняет сообщение в базе данных"""
    # Получаем текущий диалог для чата
    dialogue_id = await get_current_dialogue(chat_id)
    # Считаем количество токенов в сообщении
    token_count = await count_tokens(message)
    query = """
            INSERT INTO messages (dialogue_id, message, role, token_count)
            VALUES (%s, %s, %s, %s) \
            """
    await execute_query(query, (dialogue_id, message, role, token_count))


async def get_current_messages(chat_id):
    """Получает текущие сообщения для чата"""
    query = """
            SELECT messages.message, messages.role, messages.token_count
            FROM messages
            LEFT JOIN dialogues ON messages.dialogue_id = dialogues.id
            WHERE dialogues.chat_id = %s
              AND dialogues.is_current = TRUE
              AND message.summarized = 0 
            ORDER BY messages.time ASC \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, (chat_id,))
            return await cursor.fetchall()
