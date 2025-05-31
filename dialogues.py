from characters import get_character
from db import *


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
    await create_new_dialogue(chat_id)
    return await get_current_dialogue(chat_id)


async def create_new_dialogue(chat_id):
    # Получаем персонажа
    character = await get_character(chat_id)
    query = """
            INSERT INTO dialogues (chat_id, character_id) \
            VALUES (%s, %s) \
            """
    await execute_query(query, (chat_id, character["id"]))
