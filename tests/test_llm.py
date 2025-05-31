import os
import sys
from unittest.mock import AsyncMock

import pytest

import summarize

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import llm


def test_clean_llm_response():
    response = "Это тестовый ответ. С уважением, Арсен."
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ. С уважением, Арсен."

    response = "Это тестовый ответ *** С уважением, Арсен."
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ С уважением, Арсен."

    response = "Арсен: Это тестовый ответ"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ"

    response = "Ответ &amp; ответ"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Ответ & ответ"

    response = "Ответ &amp;#34; ответ"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == 'Ответ " ответ'

    response = "Ответ &#34; ответ"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == 'Ответ " ответ'

    response = "ой.&lt;/s&gt; *"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "ой. *"

    response = "Привет [ой] <ой>."
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Привет ой ой."

    response = "Привет! Я тебя не заметил.\n\nАрсен: Привет, Вася! Как дела?"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Привет! Я тебя не заметил."

    response = (
        "*улыбается* Ой. "
        "Просто *да*\n\n"
        "Незнакомец: ...\n\n"
        "Арсен: *кивает* "
        "*...*"
    )
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "*улыбается* Ой. Просто *да*"

    response = (
        "*улыбается* Привет! **ОБЪЯСНЕНИЕ:**\n"
        "- 123\n"
        "- 456"
    )
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "*улыбается* Привет!"

    response = (
        "*улыбается* Привет! *ОБЪЯСНЕНИЕ:*\n"
        "- 123\n"
        "- 456"
    )
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "*улыбается* Привет!"

    response = "*привет* Привет (Нет) Да. (Сценарий продолжается с"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "*привет* Привет (Нет) Да."

    response = "Это тестовое предложение\n*"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Это тестовое предложение"

    response = "Это тестовое предложение ** вот так **"
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "Это тестовое предложение * вот так *"

    response = "*это* Полностью... Нормальный, да. *ответ* ...да. да."
    cleaned_response = llm._clean_llm_response(response)
    assert cleaned_response == "*это* Полностью... Нормальный, да. *ответ* ...да. да."


def test_get_system_prompt():
    # Test with default character name
    system_prompt = llm._get_reply_system_prompt(["Абдулла поше попить", "Ты - Арсен"], "Арсен", "Класный персонаж",
                                                 "Первое резюме")
    assert "Абдулла поше попить" in system_prompt.get('content')
    assert "Ты - Арсен" in system_prompt.get('content')
    assert "Класный персонаж" in system_prompt.get('content')
    assert "Первое резюме" in system_prompt.get('content')
    assert system_prompt.get("role") == "system"


def test_trim_incomplete_sentence():
    assert llm._trim_incomplete_sentence("Привет! Как дела?") == "Привет! Как дела?"
    assert llm._trim_incomplete_sentence("Это тестовое предложение") == "Это тестовое предложение"
    assert llm._trim_incomplete_sentence("Привет. Как дела") == "Привет."
    assert llm._trim_incomplete_sentence("Тест... Что дальше") == "Тест..."
    assert llm._trim_incomplete_sentence("Без знаков окончания") == "Без знаков окончания"
    assert llm._trim_incomplete_sentence("Текст заканчивается. на *вот это*") == "Текст заканчивается. на *вот это*"
    assert llm._trim_incomplete_sentence("Текст заканчивается. на [вот это]") == "Текст заканчивается. на [вот это]"
    assert llm._trim_incomplete_sentence("Текст заканчивается. на вот это[") == "Текст заканчивается."


@pytest.mark.asyncio
async def test_build_messages(monkeypatch):
    # mock get_current_messages
    monkeypatch.setattr(llm, "get_current_messages", AsyncMock(return_value=[
        {"message": "Hello", "role": "user", "token_count": 5},
        {"message": "Hi there", "role": "assistant", "token_count": 6}
    ]))
    monkeypatch.setattr(llm, "find_similar", AsyncMock(return_value=["Арсен: Similar message"]))
    monkeypatch.setattr(llm, "get_character", AsyncMock(
        return_value={"name": "Арсен", "card": "Character card", "first_summary": "First summary"}))
    monkeypatch.setattr(llm, "get_character_name", AsyncMock(return_value="Арсен"))
    monkeypatch.setattr(summarize, "needs_summarization", AsyncMock(return_value=False))
    monkeypatch.setattr(summarize, "get_summary", AsyncMock(return_value="Summary of messages"))

    messages = await llm._build_messages(123, "test query")

    assert len(messages) == 3  # системное сообщение + 2 сообщения из истории
    assert messages[0]["role"] == "system"
    assert "Similar message" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there"
