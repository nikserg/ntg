import asyncio
import logging

import aiomysql

from config import db_config


async def get_db_connection(retries=3, delay=1):
    """Асинхронное подключение к базе данных MySQL с поддержкой повторных попыток"""
    for attempt in range(1, retries + 1):
        try:
            return await aiomysql.connect(
                host=db_config["host"],
                user=db_config["user"],
                password=db_config["password"],
                db=db_config["database"],
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
