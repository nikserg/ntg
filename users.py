import uuid

from db import get_db_connection


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


async def get_additional_individual_limit(chat_id):
    """Получает дополнительный индивидуальный лимит для пользователя"""
    query = "SELECT additional_individual_limit FROM users WHERE chat_id = %s"
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            return result[0] if result else 0
