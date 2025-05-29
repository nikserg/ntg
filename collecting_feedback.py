import logging
import re

import db
from config import ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK, ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK, \
    SUBSCRIBE_INVITE

# Хранилище для отслеживания сбора отзывов
collecting_feedback = {}


async def write_feedback(chat_id, feedback_text):
    """Сохраняет отзыв пользователя в базе данных"""
    query = """
                    INSERT INTO feedbacks (chat_id, feedback)
                    VALUES (%s, %s)
                    """
    await db.execute_query(query, (chat_id, feedback_text))


async def collect_feedback(chat_id, text):
    """Собирает отзыв от пользователя и сохраняет его в БД."""
    feedback_text = text.strip()
    if not feedback_text:
        return "Отзыв не может быть пустым. Попробуйте снова.", False
    # Проверяем, не превышает ли отзыв максимальную длину
    if len(feedback_text) > 2000:
        return "Отзыв слишком длинный. Пожалуйста, сократите его до 2000 символов.", False
    # Проверяем минимальную длину отзыва
    if len(feedback_text) < 10:
        return "Отзыв слишком короткий. Пожалуйста, напишите что-нибудь более развернутое.", False
    # Проверяем фидбек на мусорность: если в нем слишком много повторяющихся символов (типа "Это печка!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" или "аааааааааааааааааааа")
    if re.search(r'(.)\1{9,}', feedback_text):
        return "Похоже, в вашем отзыве слишком много повторяющихся символов. Пожалуйста, напишите что-то более осмысленное.", False

    try:
        await write_feedback(chat_id, feedback_text)
        return f"Спасибо за ваш отзыв! Мы ценим ваше мнение. Вы можете оставить новый отзыв в любое время с помощью команды /feedback. А сейчас вы можете продолжить чат.\n\n{SUBSCRIBE_INVITE}", True
    except Exception as e:
        logging.error(f"Ошибка при записи отзыва в БД: {e}")
        return "Не удалось сохранить отзыв. Попробуйте позже.", False


def handle_command(chat_id):
    """Обработчик команды /feedback для начала сбора отзыва."""
    collecting_feedback[chat_id] = True  # Устанавливаем статус сбора отзыва
    return (
        f"Бот сейчас в бета-версии, поэтому мы особенно ценим отзывы. За отзыв вы можете получить от "
        f"{ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK} до {ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK} сообщений в день. "
        f"Пожалуйста, напишите, что вам понравилось, что можно улучшить или какие ошибки вы заметили."
    )


async def handle_feedback(chat_id, feedback):
    """Обработчик для сбора отзывов от пользователей."""
    logging.info(f"Сбор отзыва от {chat_id}: {feedback}")
    response, successful_feedback = await collect_feedback(chat_id, feedback)
    if successful_feedback:
        collecting_feedback[chat_id] = False  # Сбрасываем статус сбора отзыва
    return response


def is_collecting_feedback(chat_id):
    """Проверяет, собирает ли бот отзывы для данного чата."""
    return collecting_feedback.get(chat_id, False)
