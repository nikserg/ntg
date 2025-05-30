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

    await fix_dialogues_chat_id()


async def fix_dialogues_chat_id():
    # Создаем столбец chat_id в таблице dialogues, если его нет
    try:
        await execute_query("""
            ALTER TABLE dialogues ADD COLUMN chat_id BIGINT NULL;
        """)
    except Exception as e:
        # Игнорируем ошибку, если столбец уже существует
        logging.info(f"При добавлении столбца chat_id в таблицу dialogues: {e}")
    # Переносим chat_id из messages в dialogues
    await execute_query("""
        UPDATE dialogues d
        JOIN messages m ON d.id = m.dialogue_id
        SET d.chat_id = m.chat_id
        WHERE d.chat_id IS NULL
    """)
    # Удаляем некорректный индекс idx_dialogues_chat_current, если он существует
    try:
        await execute_query("""
            ALTER TABLE dialogues DROP INDEX idx_dialogues_chat_current;
        """)
    except Exception as e:
        # Игнорируем ошибку, если индекс не существует
        logging.info(f"При удалении индекса idx_dialogues_chat_current: {e}")

    # Создаем индекс для chat_id в таблице dialogues
    try:
        await execute_query("""
            CREATE INDEX idx_dialogues_chat_is_current ON dialogues (chat_id, is_current);
        """)
    except Exception as e:
        # Игнорируем ошибку, если индекс уже существует
        logging.info(f"При создании индекса idx_dialogues_chat: {e}")

async def create_dialogues_migration():
    """Создает таблицу диалогов и переносит данные из таблицы сообщений."""
    # 1. Создаем таблицу диалогов
    await execute_query("""
        CREATE TABLE IF NOT EXISTS dialogues (
            id INT AUTO_INCREMENT PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            is_current BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_dialogues_chat_current (chat_id, is_current)
        );
    """)

    # 2. Получаем уникальные пары chat_id и is_current из сообщений
    await execute_query("""
        INSERT INTO dialogues (chat_id, is_current)
        SELECT DISTINCT chat_id, is_current FROM messages;
    """)

    # 3. Добавляем столбец dialogue_id в таблицу messages
    try:
        await execute_query("""
            ALTER TABLE messages ADD COLUMN dialogue_id INT NULL,
            ADD INDEX idx_messages_dialogue_id (dialogue_id);
        """)
    except Exception as e:
        logging.info(f"При добавлении столбца dialogue_id: {e}")

    # 4. Обновляем сообщения, чтобы они указывали на соответствующие диалоги
    await execute_query("""
        UPDATE messages m
        JOIN dialogues d ON m.chat_id = d.chat_id AND m.is_current = d.is_current
        SET m.dialogue_id = d.id;
    """)

    # Удаляем индекс idx_messages_chat_current, так как он больше не нужен
    try:
        await execute_query("""
            ALTER TABLE messages DROP INDEX idx_messages_chat_current;
        """)
    except Exception as e:
        logging.info(f"При удалении индекса idx_messages_chat_current: {e}")

    # Удаляем индекс idx_messages_chat_time, так как он больше не нужен
    try:
        await execute_query("""
            ALTER TABLE messages DROP INDEX idx_messages_chat_time;
        """)
    except Exception as e:
        logging.info(f"При удалении индекса idx_messages_chat_time: {e}")

    # 5. Удаляем столбец is_current из таблицы messages
    try:
        await execute_query("""
            ALTER TABLE messages DROP COLUMN is_current;
        """)
    except Exception as e:
        logging.info(f"При удалении столбца is_current: {e}")

    # Удаляем chat_id из таблицы dialogues, так как он больше не нужен - вместо него используется dialogue_id в messages
    try:
        await execute_query("""
            ALTER TABLE dialogues DROP COLUMN chat_id;
        """)
    except Exception as e:
        logging.info(f"При удалении столбца chat_id из таблицы dialogues: {e}")

    # Создаем индекс для dialogue_id и time в таблице messages
    try:
        await execute_query("""
            CREATE INDEX idx_messages_dialogue_time ON messages (dialogue_id, time);
        """)
    except Exception as e:
        logging.info(f"При создании индекса idx_messages_dialogue_time: {e}")
