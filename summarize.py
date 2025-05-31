import logging

import aiomysql

from characters import get_character
from config import SUMMARIZE_BUFFER_PERCENT, CONTEXT_TOKEN_LIMIT
from db import execute_query, get_db_connection
from dialogues import get_current_dialogue
from tokenizer import count_tokens


async def get_summarize_buffer(messages):
    """
    Разделяет сообщения на те, которые нужно пересказать, и те, которые останутся без пересказа.
    """
    tokens_to_summarize = CONTEXT_TOKEN_LIMIT * (SUMMARIZE_BUFFER_PERCENT / 100)
    logging.info(f"tokens_to_summarize: {tokens_to_summarize}")
    messages_to_summarize = []
    total_tokens = 0
    for message in messages:
        message_tokens = message["token_count"]
        if message_tokens == 0:
            message_tokens = await count_tokens(message["message"])
        total_tokens += message_tokens
        messages_to_summarize.append(message)
        if total_tokens >= tokens_to_summarize:
            break
    messages_without_summarize = messages[len(messages_to_summarize):]
    return messages_to_summarize, messages_without_summarize


async def needs_summarization(messages):
    """
    Проверяет, нужно ли пересказывать сообщения.
    Возвращает True, если количество токенов в сообщениях превышает лимит контекста.
    """
    total_tokens = await _count_tokens_in_messages(messages)
    return total_tokens > CONTEXT_TOKEN_LIMIT


async def truncate_history_overflow(messages):
    return await _truncate_history(messages, context_token_limit=CONTEXT_TOKEN_LIMIT * 1.5)


async def _truncate_history(messages, context_token_limit=CONTEXT_TOKEN_LIMIT):
    """
    Обрезает историю сообщений, чтобы оставить максимальное
    количество последних сообщений в рамках max_tokens.
    """
    if not messages:
        return []

    # Подсчёт токенов для каждого сообщения
    total_tokens = 0
    truncated = []

    for msg in reversed(messages):
        msg_tokens = msg["token_count"]
        if msg_tokens is None or msg_tokens <= 0:
            # Если токены не посчитаны, используем функцию для подсчёта
            msg_tokens = await count_tokens(msg["message"])

        if total_tokens + msg_tokens > context_token_limit:
            break
        truncated.insert(0, msg)
        total_tokens += msg_tokens

    return truncated


async def _count_tokens_in_messages(messages):
    """
    Считает общее количество токенов в списке сообщений.
    """
    total_tokens = 0
    for message in messages:
        message_tokens = message["token_count"]
        if message_tokens == 0:
            message_tokens = await count_tokens(message["message"])
        total_tokens += message_tokens
    return total_tokens


async def mark_messages_as_summarized(chat_id, messages):
    """
    Помечает сообщения как пересказанные в базе данных.
    """
    # Получаем самое новое пересказанное сообщение (с наибольшим id)
    max_id = max(msg["id"] for msg in messages)
    logging.info(f"Максимальный ID сообщения для пометки как пересказанное: {max_id}")
    dialogue_id = await get_current_dialogue(chat_id)
    query = f"UPDATE messages SET summarized = 1 WHERE dialogue_id = %s AND id <= %s"
    await execute_query(query, (dialogue_id, max_id))


async def write_summary_to_db(chat_id, summary):
    """
    Записывает пересказ в базу данных.
    """
    try:
        tokens = await count_tokens(summary)
        dialogue_id = await get_current_dialogue(chat_id)
        query = "INSERT INTO summaries (dialogue_id, summary, token_count) VALUES (%s, %s, %s)"
        await execute_query(query, (dialogue_id, summary, tokens))
    except Exception as e:
        logging.error("Ошибка при записи пересказа в базу данных: %s", e)
        raise e


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
    try:
        await write_summary_to_db(chat_id, summary)
    except Exception as e:
        logging.error("Ошибка при сохранении первого пересказа в базу данных: %s", e)
        raise e
    return summary
