import aiomysql

from db import get_db_connection


async def get_character(chat_id):
    """Получает первого персонажа из базы данных"""
    query = "SELECT * FROM characters LIMIT 1"
    async with (await get_db_connection()) as conn:
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query)
            character = await cursor.fetchone()
            if character:
                return character
    raise RuntimeError(f"Не удалось получить персонажа для чата {chat_id}")


async def get_character_name(chat_id):
    """Получает имя персонажа для чата"""
    character = await get_character(chat_id)
    return character["name"]
