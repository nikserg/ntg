import logging

from db import execute_query


async def apply_migrations():
    """Применяет миграции для базы данных."""
    await execute_query("""
        CREATE TABLE IF NOT EXISTS messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            message TEXT NOT NULL,
            role VARCHAR(50) NOT NULL,
            is_current BOOLEAN DEFAULT TRUE,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    await execute_query("""
        CREATE TABLE IF NOT EXISTS feedbacks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            feedback TEXT NOT NULL,
            time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    await execute_query("""
        CREATE TABLE IF NOT EXISTS users (
            chat_id BIGINT PRIMARY KEY,
            invite_code VARCHAR(36) UNIQUE NOT NULL,
            invited_by BIGINT DEFAULT NULL,
            registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX (invited_by)
        );
    """)

    # Добавляем индексы отдельно, чтобы не вызывать ошибки при повторном запуске
    add_indexes = [
        "ALTER TABLE messages ADD INDEX idx_messages_chat_current (chat_id, is_current);",
        "ALTER TABLE messages ADD INDEX idx_messages_chat_time (chat_id, time);",
        "ALTER TABLE feedbacks ADD INDEX idx_feedbacks_chat_id (chat_id);"
    ]
    for index_query in add_indexes:
        try:
            await execute_query(index_query)
        except Exception as e:
            # Игнорируем ошибки, если индекс уже существует
            logging.info(f"При создании индекса: {e}")
    # Добавляем столбец за полезный фидбек, если его нет
    alter_feedbacks_table = """
        ALTER TABLE feedbacks ADD COLUMN useful BOOLEAN DEFAULT FALSE;
    """
    try:
        await execute_query(alter_feedbacks_table)
    except Exception as e:
        # Игнорируем ошибки, если столбец уже существует
        logging.info(f"При добавлении столбца useful в таблицу feedbacks: {e}")

    # Добавляем хранилище изображений
    await execute_query("""
        CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content MEDIUMBLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
