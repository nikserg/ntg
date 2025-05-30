import asyncio
import logging

import aiohttp

from config import TOKENIZER_ENDPOINT


async def count_tokens(text):
    # Обращаемся к отдельному сервису для токенизации, повторяем попытки при ошибках
    async with aiohttp.ClientSession() as session:
        retries = 3
        for attempt in range(retries):
            try:
                async with session.post(TOKENIZER_ENDPOINT, json={"text": text}, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
                    msg_tokens = data.get("tokens", 0)
                    break
            except Exception as e:
                logging.error(f"Ошибка при токенизации сообщения: {e}")
                if attempt == retries - 1:
                    msg_tokens = len(text)  # Если не удалось получить токены, используем длину сообщения
                await asyncio.sleep(2 * (attempt + 1))  # экспоненциальная задержка
    return msg_tokens
