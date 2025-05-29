import logging

from db import get_db_connection


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
