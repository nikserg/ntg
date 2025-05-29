import asyncio
import logging
import re

import aiohttp

import qdrant
from config import RUNPOD_API_KEY, CHARACTER_NAME, TOKENIZER_ENDPOINT, RUNPOD_ENDPOINT, SYSTEM_PROMPT, CHARACTER_CARD, \
    TEMPERATURE, TOP_P, MIN_P, REPEAT_PENALTY, REPLY_MAX_TOKENS, CONTEXT_TOKEN_LIMIT
from db import get_current_messages


def trim_incomplete_sentence(text):
    # Находит последнее завершённое предложение
    match = re.search(r'([.!?…\]*])[^.!?…\]*]*$', text)
    if match:
        end = match.end(1)
        return text[:end].strip()
    return text.strip()


def clean_llm_response(text):
    """Удаляет нежелательные символы или строки из ответа."""
    text = text.replace("***", "").strip()
    # Удаляем строку, начинающуюся с имени персонажа
    if text.startswith(f"{CHARACTER_NAME}:"):
        text = text[len(f"{CHARACTER_NAME}:"):].strip()
    # Обрезаем, если ответ заканчивается на '\n\nИмя:'
    text = re.split(r'\n\w+:', text)[0]
    # Обрезаем, если в конце есть объясниение
    text = re.split(r'\*+ОБЪЯСНЕНИЕ:', text)[0]
    # Обрезаем, если в сообщении есть упоминание развития сценария
    text = re.split(r'\(Сценарий ', text)[0]
    # Заменяем двойные пробелы на одинарные
    text = re.sub(r'\s{2,}', ' ', text)
    # Заменяем последовательность &amp; на амперсанд
    text = text.replace("&amp;", "&")
    # Заменяем последовательность &#34; на кавычки
    text = text.replace("&#34;", '"')
    # Заменяем HTML-сущности на соответствующие символы
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    # Убираем теги HTML
    text = re.sub("<s>", '', text)
    text = re.sub("</s>", '', text)
    # Убираем квадратные скобки
    text = re.sub(r"\[", '', text)
    text = re.sub("]", '', text)
    # Убираем угловые скобки
    text = re.sub("<", '', text)
    text = re.sub(">", '', text)
    # Убираем неоконченные предложения
    text = trim_incomplete_sentence(text)
    return text.strip()


async def run_llm(chat_id, cleaned_input):
    messages = await build_messages(chat_id, cleaned_input)
    payload = {
        "input": {
            "messages": messages,
            "params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "min_p": MIN_P,
                "repeat_penalty": REPEAT_PENALTY,
                "max_tokens": REPLY_MAX_TOKENS
            }
        }
    }
    logging.info(f"Отправка запроса к LLM: {payload}")
    retries = 3  # Количество попыток при ошибках
    delay = 2  # Начальная задержка между попытками
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                async with session.post(RUNPOD_ENDPOINT + "/runsync", json=payload, headers=headers,
                                        timeout=240) as response:
                    response.raise_for_status()
                    data = await response.json()
                    # Проверяем, не вернулся ли статус IN_PROGRESS
                    if data.get("status") in ["IN_PROGRESS", "IN_QUEUE"] and data.get("id"):
                        task_id = data.get("id")
                        logging.info(f"Получен статус {data.get('status')}, id задачи: {task_id}")

                        # Ждем завершения задачи
                        max_attempts = 120
                        attempts = 0

                        while attempts < max_attempts:
                            await asyncio.sleep(1)
                            async with session.get(f"{RUNPOD_ENDPOINT}/status/{task_id}",
                                                   headers=headers, timeout=240) as status_response:
                                status_data = await status_response.json()
                                status = status_data.get("status")

                                if status == "COMPLETED":
                                    output = status_data.get("output", {})
                                    text = output.get("text")
                                    if text is None:
                                        logging.error(f"Ответ LLM без текста: {status_data}")
                                        return "[Ой! Кажется, у меня техническая проблема под кодовым именем Клубничка]"
                                    return clean_llm_response(text)

                                elif status == "FAILED":
                                    logging.info(f"Задача завершилась с ошибкой: {status_data}")
                                    return "[Ой! Кажется, у меня техническая проблема под кодовым именем Апельсин]"

                                # Иначе продолжаем ожидание

                        logging.error(f"Превышено время ожидания для задачи {task_id}")
                        return "[Ой! Кажется, у меня техническая проблема под кодовым именем Лимон]"

                    # Обычная обработка для случая немедленного ответа
                    output = data.get("output", {})
                    text = output.get("text")
                    if text is None:
                        logging.error(f"Ответ LLM без текста: {data}")
                        return "[Ой! Кажется, у меня техническая проблема под кодовым именем Клубничка]"
                    return clean_llm_response(text)
        except Exception as e:
            logging.error(f"Ошибка при обращении к RunPod (попытка {attempt}): {e}")
            if attempt == retries:
                return "[Ой! Кажется, у меня техническая проблема под кодовым именем Персик]"
            await asyncio.sleep(delay)
            delay *= 2  # экспоненциальная задержка
    return None


async def truncate_history(messages, max_tokens):
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
        # Обращаемся к отдельному сервису для токенизации, повторяем попытки при ошибках
        async with aiohttp.ClientSession() as session:
            retries = 3
            for attempt in range(retries):
                try:
                    async with session.post(TOKENIZER_ENDPOINT, json={"text": msg["message"]}, timeout=30) as response:
                        response.raise_for_status()
                        data = await response.json()
                        msg_tokens = data.get("tokens", 0)
                        break
                except Exception as e:
                    logging.error(f"Ошибка при токенизации сообщения: {e}")
                    if attempt == retries - 1:
                        msg_tokens = len(msg["message"])  # Если не удалось получить токены, используем длину сообщения
                    await asyncio.sleep(2 * (attempt + 1))  # экспоненциальная задержка

        if total_tokens + msg_tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total_tokens += msg_tokens

    return truncated


async def build_messages(chat_id, user_input):
    """
    Формирует список сообщений для LLM из истории чата, векторной БД и системного промпта.
    """
    # Получаем историю чата из MySQL
    history_records = await get_current_messages(chat_id)

    # Обрезаем историю до лимита токенов
    history = await truncate_history(history_records, CONTEXT_TOKEN_LIMIT)

    # Находим похожие сообщения из векторной БД
    memories = await qdrant.find_similar(user_input, chat_id, current_context=[msg["message"] for msg in history])

    # Формируем системное сообщение
    system_message = {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n***\n{CHARACTER_CARD}" + (
            f"\n***\nПредыдущие сообщения:\n" + "\n".join(memories) if memories else "")
    }

    # Формируем сообщения из истории чата
    messages = [system_message]

    for msg in history:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["message"]})
        else:
            messages.append({"role": "user", "content": msg["message"]})

    return messages
