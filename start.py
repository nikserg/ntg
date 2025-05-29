import logging

import qdrant
from config import FIRST_MESSAGE, USER_FIRST_MESSAGE
from db import get_db_connection, save_message, execute_query
from db import get_or_create_user
from sanitizer import remove_newlines


async def reset_history_in_db(chat_id):
    """Сбрасывает историю сообщений для чата, помечая все сообщения как неактуальные"""
    query = """
                UPDATE messages
                SET is_current = FALSE
                WHERE chat_id = %s \
                """
    await execute_query(query, (chat_id))


# Асинхронный сброс истории
async def reset_history(chat_id):
    # Сбрасываем историю сообщений в БД
    await reset_history_in_db(chat_id)
    logging.info(f"История сообщений для chat_id={chat_id} успешно сброшена в БД.")

    # Удаление данных из Qdrant
    try:
        qdrant.qdrant_delete(chat_id)
        logging.info(f"Данные для chat_id={chat_id} успешно удалены из Qdrant.")
    except Exception as e:
        logging.error(f"Ошибка при удалении данных из Qdrant для chat_id={chat_id}: {e}")


async def handle_command(chat_id, user_input):
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
    await get_or_create_user(chat_id, invited_by)

    # Сбрасываем историю сообщений и отправляем приветственное сообщение
    await reset_history_in_db(chat_id)
    return await add_first_messages_to_db(chat_id)


async def add_first_messages_to_db(chat_id):
    first_message = FIRST_MESSAGE.replace('\\n', '\n')
    await save_message(chat_id, USER_FIRST_MESSAGE,
                       "user")  # Это сообщение не будет видно пользователю, но сохранится в БД
    await save_message(chat_id, remove_newlines(first_message), "assistant")
    return FIRST_MESSAGE.replace("\\n", "\n")
