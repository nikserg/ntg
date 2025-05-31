import asyncio
import logging
import re

import aiohttp

import summarize
from characters import get_character, get_character_name
from config import RUNPOD_API_KEY, RUNPOD_ENDPOINT, SYSTEM_PROMPT, \
    TEMPERATURE, TOP_P, MIN_P, REPEAT_PENALTY, REPLY_MAX_TOKENS, \
    SUMMARIZE_TARGET_TOKEN_LENGTH, SUMMARIZE_TEMPERATURE
from messages import get_current_messages
from qdrant import find_similar


async def run_llm(chat_id, cleaned_input):
    """Получает ответ от LLM на основе входящего сообщения пользователя."""
    try:
        messages = await _build_messages(chat_id, cleaned_input)
        character_name = await get_character_name(chat_id)
        llm_response = await _llm_request(messages, character_name=character_name, include_character_name=True)
        return _clean_llm_response(llm_response)
    except Exception as e:
        logging.error(f"Ошибка при запросе к LLM: {e}.")
        raise e


async def _build_messages(chat_id, user_input):
    """
    Формирует список сообщений для LLM из истории чата, векторной БД и системного промпта.
    """
    # Получаем персонажа
    character = await get_character(chat_id)

    # Получаем историю чата из MySQL
    history = await get_current_messages(chat_id)

    # Если сообщения начинают обрезаться, делаем пересказ
    summary = await summarize.get_summary(chat_id)
    if await summarize.needs_summarization(history):
        # Получаем пересказ и обрезанную историю сообщений
        summary, history = await _make_new_summary(summary, chat_id, history, character.get("name", ""))

    # Находим похожие сообщения из векторной БД
    memories = await find_similar(user_input, chat_id, current_context=[msg["message"] for msg in history])

    # Формируем системное сообщение для ответа в чате
    chat_system_prompt = _get_reply_system_prompt(memories, character.get("name"), character.get("card"), summary)

    # Формируем сообщения из истории чата
    return _make_messages_with_system_prompt(chat_system_prompt, history)


async def _make_new_summary(previous_summary, chat_id, message_history, character_name):

    # Выводим ID сообщений, которые будут пересказаны
    logging.info(f"Создание пересказа для {chat_id}: {[msg['id'] for msg in message_history]}")

    # Если сообщений для пересказа слишком много (после запуска новой системы), обрезаем историю
    # Увеличиваем лимит контекста, чтобы туда поместились граничные сообщения, которые могут выпасть из
    # контекста, особенно при первом пересказе
    message_history = await summarize.truncate_history_overflow(message_history)
    logging.info(f"История для пересказа {chat_id} после обрезания: {[msg['id'] for msg in message_history]}")

    # Сообщения из буфера для пересказа объединяем в одно сообщение
    history_to_summarize, history_without_summarized = await summarize.get_summarize_buffer(message_history)
    logging.info(f"Сообщения для пересказа: {[msg['id'] for msg in history_to_summarize]}")
    logging.info(f"Сообщения, которые не будут пересказаны: {[msg['id'] for msg in history_without_summarized]}")

    user_message = _collapse_history_to_single_message(history_to_summarize, previous_summary, character_name)
    system_prompt = {
        "role": "system",
        "content": (
            "Составь краткий пересказ этого текста от третьего лица, упоминая имена персонажей, их внешность (если известна), затронутые в разговоре темы (если таковые есть). "
            "Избегай прямого цитирования, используй только пересказ и краткие описания. Сосредоточься на ключевых событиях и сухих фактах. "
            "Не добавляй собственные эмоции, мнения или оценки происходящего. "
        )
    }
    summarize_messages_request = _make_messages_with_system_prompt(system_prompt,
                                                                   [{"role": "user", "message": user_message}])
    # Запрос к LLM для пересказа
    try:
        summary = await _llm_request(
            messages=summarize_messages_request,
            character_name=character_name,
            include_character_name=False,
            max_tokens=SUMMARIZE_TARGET_TOKEN_LENGTH,
            temperature=SUMMARIZE_TEMPERATURE
        )
        # Очищаем ответ от лишних символов
        summary = _clean_llm_summary(summary)
    except Exception as e:
        logging.error(f"Ошибка при запросе пересказа в LLM: {e}")
        raise e

    logging.info(
        (f"История сообщений для чата {chat_id} обрезана. "
         f"Сообщений для пересказа {len(history_to_summarize)}. "
         f"Температура {SUMMARIZE_TEMPERATURE}. "
         f"Токены {SUMMARIZE_TARGET_TOKEN_LENGTH}.\n"
         f"Запрос: {summarize_messages_request}.\n"
         f"Ответ LLM: {summary}"
         )
    )
    # Сохраняем пересказ в БД
    await summarize.write_summary_to_db(chat_id, summary)
    # Помечаем сообщения в БД как пересказанные
    await summarize.mark_messages_as_summarized(chat_id, history_to_summarize)
    return summary, history_without_summarized


def _collapse_history_to_single_message(messages, previous_summary, character_name):
    """
    Формирует диалог в одно большое сообщение для пересказа.
    """
    user_message = f"{previous_summary}\n***\n"
    for msg in messages:
        if msg["role"] == "user":
            user_message += f"Собеседник: {msg['message']}\n"
        elif msg["role"] == "assistant":
            user_message += f"{character_name}: {msg['message']}\n"
    message = user_message.strip()
    return message.strip()


def _make_messages_with_system_prompt(system_prompt, chat_messages):
    if system_prompt:
        messages = [system_prompt]
    else:
        messages = []

    for msg in chat_messages:
        if msg["role"] == "assistant":
            messages.append({"role": "assistant", "content": msg["message"]})
        else:
            messages.append({"role": "user", "content": msg["message"]})

    return messages


def _get_reply_system_prompt(memories, character_name, character_card, summary):
    """Возвращает системный промпт для LLM."""
    return {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n***\nТвой персонаж: {character_name}\n{character_card}\n***\nСюжет:\n{summary}" + (
            f"\n***\nПредыдущие сообщения:\n" + "\n".join(memories) if memories else "")
    }


def _clean_llm_summary(text):
    # Убираем неоконченные предложения
    text = _trim_incomplete_sentence(text)
    # Заменяем переносы строк на пробелы
    text = re.sub(r'\n+', ' ', text)
    # Убираем лишние пробелы и переносы строк
    text = re.sub(r'\s+', ' ', text).strip()
    # Убираем пробелы в начале и конце
    text = text.strip()
    return text

def _clean_llm_response(text):
    """Удаляет нежелательные символы или строки из ответа."""
    text = text.replace("***", "").strip()
    # Удаляем строку, начинающуюся с имени
    text = re.sub(r'^\w+:\s*', '', text)
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
    text = re.sub(r"\n\*$", '', text)

    # Убираем неоконченные предложения
    text = _trim_incomplete_sentence(text)
    return text.strip()


def _trim_incomplete_sentence(text):
    # Находит последнее завершённое предложение
    match = re.search(r'([.!?…\]*])[^.!?…\]*]*$', text)
    if match:
        end = match.end(1)
        return text[:end].strip()
    return text.strip()

async def _llm_request(messages, character_name, include_character_name=True, temperature=TEMPERATURE, top_p=TOP_P,
                       min_p=MIN_P, repeat_penalty=REPEAT_PENALTY,
                       max_tokens=REPLY_MAX_TOKENS):
    """ Отправляет запрос к LLM и получает ответ."""
    payload = {
        "input": {
            "messages": messages,
            "character_name": character_name,
            "include_character_name": include_character_name,
            "params": {
                "temperature": temperature,
                "top_p": top_p,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "max_tokens": max_tokens
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
                                    return text

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
                    return text
        except Exception as e:
            logging.error(f"Ошибка при обращении к RunPod (попытка {attempt}): {e}")
            if attempt == retries:
                return "[Ой! Кажется, у меня техническая проблема под кодовым именем Персик]"
            await asyncio.sleep(delay)
            delay *= 2  # экспоненциальная задержка
    return None
