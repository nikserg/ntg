from config import ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED, BOT_NAME, SUBSCRIBE_INVITE
from db import get_or_create_user


async def handle_command(chat_id):
    """Обработчик команды /invite для получения инвайт-кода."""
    invite_code = await get_or_create_user(chat_id)
    invite_link = f"https://t.me/{BOT_NAME}?start={invite_code}"

    return (
        f"Поделитесь этой ссылкой с друзьями и получите дополнительно "
        f"{ADDITIONAL_MESSAGES_PER_DAY_FOR_INVITED} сообщений в день "
        f"за каждого приглашённого пользователя:\n\n{invite_link}"
        f"\n\n{SUBSCRIBE_INVITE}"
    )
