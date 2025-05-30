import base64
import logging

import qdrant
from characters import get_character
from db import get_db_connection, execute_query
from messages import save_message
from sanitizer import remove_newlines
from users import get_or_create_user


async def reset_history_in_db(chat_id):
    """Сбрасывает историю сообщений для чата, помечая все сообщения как неактуальные"""
    query = """
                UPDATE dialogues \
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

    # Сбрасываем историю сообщений в БД и Qdrant
    await reset_history(chat_id)

    # Возвращаем текст ответа и картинку
    return await add_first_messages_to_db(chat_id), await get_random_start_image()


async def add_first_messages_to_db(chat_id):
    character = await get_character(chat_id)
    first_message = character["first_message"]
    await save_message(chat_id, character["user_first_message"], "user")  # Это сообщение только сохранится в БД
    await save_message(chat_id, remove_newlines(first_message), "assistant")
    return first_message


async def get_random_start_image():
    """
    Получает случайное изображение из базы данных и возвращает его в виде base64-строки.

    Returns:
        str: base64-encoded image string or None if no images found
    """
    try:
        async with (await get_db_connection()) as conn:
            async with conn.cursor() as cursor:
                # Выбираем случайное изображение
                await cursor.execute("SELECT content FROM images ORDER BY RAND() LIMIT 1")
                result = await cursor.fetchone()

                if not result:
                    logging.warning("Изображения не найдены в базе данных")
                    return None

                # Преобразуем бинарные данные в base64
                binary_data = result[0]
                base64_data = base64.b64encode(binary_data).decode('utf-8')

                return base64_data

    except Exception as e:
        logging.error(f"Ошибка при получении случайного изображения: {e}")
        return None
