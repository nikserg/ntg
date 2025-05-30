import asyncio
import logging
import re

import aiohttp

import qdrant
from characters import get_character
from config import RUNPOD_API_KEY, RUNPOD_ENDPOINT, SYSTEM_PROMPT, \
    TEMPERATURE, TOP_P, MIN_P, REPEAT_PENALTY, REPLY_MAX_TOKENS, CONTEXT_TOKEN_LIMIT
from messages import get_current_messages
from tokenizer import count_tokens


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
    # Удаляем строку, начинающуюся с имени
    if text.startswith(r'\w+:'):
        text = text[len(r"\w+:"):].strip()
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
    # Убираем двойные звездочки
    text = re.sub(r"\*\*", '*', text)
    # Убираем одну звездочку, если она идет в конце сразу после переноса строки
    text = re.sub(r"\n\*$", '\n', text)

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
        msg_tokens = msg["token_count"]
        if msg_tokens is None or msg_tokens <= 0:
            # Если токены не посчитаны, используем функцию для подсчёта
            msg_tokens = await count_tokens(msg["message"])

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

    # Получаем персонажа
    character = await get_character(chat_id)

    # Формируем системное сообщение
    system_message = get_system_prompt(memories, character.get("name"), character.get("card"),
                                       character.get("first_summary"))

    # Формируем сообщения из истории чата
    messages = [system_message]

    for msg in history:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["message"]})
        else:
            messages.append({"role": "user", "content": msg["message"]})

    return messages


def get_system_prompt(memories, character_name, character_card, summary):
    """Возвращает системный промпт для LLM."""
    return {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n***\nТвой персонаж: {character_name}\n{character_card}\n***\nКонтекст:\n{summary}" + (
            f"\n***\nПредыдущие сообщения:\n" + "\n".join(memories) if memories else "")
    }
