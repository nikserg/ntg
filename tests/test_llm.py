import os
import sys
from unittest.mock import AsyncMock

import pytest
from aioresponses import aioresponses

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import llm
import db
import qdrant

def test_clean_llm_response():
    response = "Это тестовый ответ. С уважением, Арсен."
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ. С уважением, Арсен."

    response = "Это тестовый ответ *** С уважением, Арсен."
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ С уважением, Арсен."

    response = "Арсен: Это тестовый ответ"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ"

    response = "Ответ &amp; ответ"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Ответ & ответ"

    response = "Ответ &amp;#34; ответ"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == 'Ответ " ответ'

    response = "Ответ &#34; ответ"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == 'Ответ " ответ'

    response = "ой.&lt;/s&gt; *"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "ой. *"

    response = "Привет [ой] <ой>."
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Привет ой ой."

    response = "Привет! Я тебя не заметил.\n\nАрсен: Привет, Вася! Как дела?"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Привет! Я тебя не заметил."

    response = (
        "*улыбается* Ой. "
        "Просто *да*\n\n"
        "Незнакомец: ...\n\n"
        "Арсен: *кивает* "
        "*...*"
    )
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "*улыбается* Ой. Просто *да*"

    response = (
        "*улыбается* Привет! **ОБЪЯСНЕНИЕ:**\n"
        "- 123\n"
        "- 456"
    )
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "*улыбается* Привет!"

    response = (
        "*улыбается* Привет! *ОБЪЯСНЕНИЕ:*\n"
        "- 123\n"
        "- 456"
    )
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "*улыбается* Привет!"

    response = "*привет* Привет (Нет) Да. (Сценарий продолжается с"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "*привет* Привет (Нет) Да."

    response = "Это тестовое предложение\n*"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовое предложение"

    response = "Это тестовое предложение ** вот так **"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовое предложение * вот так *"


def test_get_system_prompt():
    # Test with default character name
    system_prompt = llm.get_system_prompt(["Абдулла поше попить", "Ты - Арсен"])
    assert "Абдулла поше попить" in system_prompt.get('content')
    assert "Ты - Арсен" in system_prompt.get('content')
    assert system_prompt.get("role") == "system"


def test_trim_incomplete_sentence():
    assert llm.trim_incomplete_sentence("Привет! Как дела?") == "Привет! Как дела?"
    assert llm.trim_incomplete_sentence("Это тестовое предложение") == "Это тестовое предложение"
    assert llm.trim_incomplete_sentence("Привет. Как дела") == "Привет."
    assert llm.trim_incomplete_sentence("Тест... Что дальше") == "Тест..."
    assert llm.trim_incomplete_sentence("Без знаков окончания") == "Без знаков окончания"
    assert llm.trim_incomplete_sentence("Текст заканчивается. на *вот это*") == "Текст заканчивается. на *вот это*"
    assert llm.trim_incomplete_sentence("Текст заканчивается. на [вот это]") == "Текст заканчивается. на [вот это]"
    assert llm.trim_incomplete_sentence("Текст заканчивается. на вот это[") == "Текст заканчивается."

@pytest.mark.asyncio
async def test_truncate_history(monkeypatch):
    from llm import truncate_history, TOKENIZER_ENDPOINT

    messages = [
        {"message": "a" * 5},
        {"message": "b" * 10},
        {"message": "c" * 20}
    ]

    with aioresponses() as m:
        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 5})
        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 10})
        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 20})

        result = await truncate_history(messages, max_tokens=100)
        assert result == messages

        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 10})
        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 20})
        result = await truncate_history(messages, max_tokens=31)
        assert result == [{"message": "b" * 10}, {"message": "c" * 20}]

        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 20})
        result = await truncate_history(messages, max_tokens=20)
        assert result == [{"message": "c" * 20}]

        m.post(TOKENIZER_ENDPOINT, payload={"tokens": 20})
        result = await truncate_history(messages, max_tokens=19)
        assert result == []


@pytest.mark.asyncio
async def test_build_messages(monkeypatch):
    # mock db.get_current_messages
    monkeypatch.setattr(db, "get_current_messages", AsyncMock(return_value=[
        {"message": "Hello", "role": "user"},
        {"message": "Hi there", "role": "assistant"}
    ]))
    monkeypatch.setattr(llm, "truncate_history", AsyncMock(side_effect=lambda msgs, max_tokens: msgs))
    monkeypatch.setattr(qdrant, "find_similar", AsyncMock(return_value=["Арсен: Similar message"]))

    messages = await llm.build_messages(123, "test query")

    assert len(messages) == 3  # системное сообщение + 2 сообщения из истории
    assert messages[0]["role"] == "system"
    assert "Similar message" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there"
