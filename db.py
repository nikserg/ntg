import asyncio
import logging

import aiomysql

from config import DB_CONFIG


async def get_db_connection(retries=3, delay=1):
    """Асинхронное подключение к базе данных MySQL с поддержкой повторных попыток"""
    for attempt in range(1, retries + 1):
        try:
            return await aiomysql.connect(
                host=DB_CONFIG["host"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                db=DB_CONFIG["database"],
                autocommit=True
            )
        except Exception as e:
            logging.error(f"Ошибка подключения к MySQL (попытка {attempt}): {e}")
            if attempt == retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2  # экспоненциальная задержка
    return None


async def execute_query(query, params=None):
    async with (await get_db_connection()) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(query, params)
