from config import (SUBSCRIBE_INVITE, ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK,
                    ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK, MAX_MESSAGES_PER_DAY,
                    ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED, BOT_NAME, ADDITIONAL_MESSAGES_PER_DAY_FOR_SUBSCRIPTION)
from db import get_db_connection, get_or_create_user
from subscribe_check import check_subscription


# Асинхронное получение количества сообщений за день
async def count_daily_messages(chat_id):
    query = """
            SELECT COUNT(*)
            FROM messages
            LEFT JOIN dialogues ON messages.dialogue_id = dialogues.id
            WHERE dialogues.chat_id = %s
              AND DATE(messages.time) = CURDATE() \
            """
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            return result[0] if result else 0


async def get_feedback_bonus(chat_id):
    """
    Возвращает кортеж (has_feedback, is_useful), где:
    - has_feedback: True, если есть хотя бы один фидбек
    - is_useful: True, если хотя бы один фидбек отмечен как полезный
    """
    query = "SELECT useful FROM feedbacks WHERE chat_id = %s ORDER BY useful DESC LIMIT 1"
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            if result is None:
                return False, False
            is_useful = bool(result[0]) if result[0] is not None else False
            return True, is_useful


async def get_invited_users_count(chat_id):
    """Возвращает количество приглашённых пользователей"""
    query = "SELECT COUNT(*) FROM users WHERE invited_by = %s"
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, (chat_id,))
            result = await cursor.fetchone()
            return result[0] if result else 0


async def get_user_daily_limit(chat_id):
    """Вычисляет дневной лимит сообщений для пользователя"""
    invited_count = await get_invited_users_count(chat_id)
    # Ограничиваем количество приглашений, учитываемых для бонуса
    bonus_count = min(invited_count, 5)
    # Проверяем наличие фидбека
    has_feedback, is_useful = await get_feedback_bonus(chat_id)
    feedback_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK if is_useful else ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK if has_feedback else 0
    # Проверяем, подписан ли пользователь на канал
    subscription_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_SUBSCRIPTION if await check_subscription(chat_id) else 0
    return MAX_MESSAGES_PER_DAY + (
                ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED * bonus_count) + feedback_bonus + subscription_bonus


async def handle_command(chat_id):
    # Получаем текущие показатели
    daily_message_count = await count_daily_messages(chat_id)
    invited_count = await get_invited_users_count(chat_id)
    bonus_count = min(invited_count, 5)
    daily_limit = await get_user_daily_limit(chat_id)

    # Получаем бонусы за фидбек
    has_feedback, is_useful = await get_feedback_bonus(chat_id)
    feedback_bonus_str = ""
    if has_feedback:
        feedback_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK if is_useful else ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK
        feedback_type = "полезный" if is_useful else "обычный"
        feedback_bonus_str = f"• Бонус за отзыв: +{feedback_bonus} ({feedback_type})"

    # Вычисляем базовый лимит и бонус от приглашений
    base_limit = MAX_MESSAGES_PER_DAY
    invite_bonus = ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED * bonus_count

    # Проверяем подписку
    is_subscribed = await check_subscription(chat_id)
    if is_subscribed:
        subscription_bonus_str = f"• Бонус за подписку: +{ADDITIONAL_MESSAGES_PER_DAY_FOR_SUBSCRIPTION}"
    else:
        subscription_bonus_str = "• Бонус за подписку: 0 (нет подписки на канал)"

    return (
        f"Ваш лимит сообщений на сегодня: {daily_limit}\n"
        f"• Базовый лимит: {base_limit}\n"
        f"• Бонус за приглашения: +{invite_bonus} ({invited_count} приглашённых, учитывается максимум 5)\n"
        f"{feedback_bonus_str}\n"
        f"{subscription_bonus_str}\n"
        f"Использовано сегодня: {daily_message_count} из {daily_limit}\n"
        f"\n"
        f"Используйте /invite для получения ссылки-приглашения и увеличения лимита.\n"
        f"Оставьте отзыв с помощью /feedback, чтобы получить дополнительный бонус!"
        f"\n\n{SUBSCRIBE_INVITE}"
    )


async def is_limit_exceeded(chat_id):
    daily_message_count = await count_daily_messages(chat_id)
    user_daily_limit = await get_user_daily_limit(chat_id)
    if daily_message_count < user_daily_limit:
        return False, None  # Лимит не превышен, ничего не возвращаем
    # Получаем инвайт-код пользователя
    invite_code = await get_or_create_user(chat_id)
    invite_link = f"https://t.me/{BOT_NAME}?start={invite_code}"

    return True, (
        f"Вы достигли лимита бета-версии в {user_daily_limit} сообщений за сегодня. "
        f"Счетчик будет сброшен завтра.\n\n"
        f"Поделитесь своим опытом с помощью команды /feedback, чтобы получить от {ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK} до {ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK} дополнительных сообщений в день!\n\n"
        f"А также вы можете пригласить друзей по этой ссылке и получить дополнительно "
        f"{ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED} сообщений в день за каждого приглашённого:\n{invite_link}\n\n"
        f"{SUBSCRIBE_INVITE}")
