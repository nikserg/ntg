from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses

import llm


def test_clean_llm_response():
    response = "Это тестовый ответ. С уважением, Арсен"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ. С уважением, Арсен"

    response = "Это тестовый ответ *** С уважением, Арсен"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ С уважением, Арсен"

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

    response = "*привет* Привет (Нет) Да (Сценарий продолжается с"
    cleaned_response = llm.clean_llm_response(response)
    assert cleaned_response == "*привет* Привет (Нет) Да"


@pytest.mark.asyncio
async def test_run_llm_error():
    # Правильно создаем моки для асинхронного контекстного менеджера
    post_ctx = AsyncMock()
    post_ctx.__aenter__ = AsyncMock(side_effect=Exception("fail"))
    post_ctx.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.post = MagicMock(return_value=post_ctx)

    client_session = AsyncMock()
    client_session.__aenter__ = AsyncMock(return_value=session)
    client_session.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=client_session):
        result = await llm.run_llm(1, "промпт")
        assert "техническая проблема" in result


@pytest.mark.asyncio
async def test_run_llm_in_progress_failed():
    """Тестирует сценарий, когда RunPod сначала возвращает IN_PROGRESS, а затем FAILED"""
    # Первый ответ со статусом IN_PROGRESS
    in_progress_response = {
        "delayTime": 46842,
        "id": "sync-task-id",
        "status": "IN_PROGRESS",
        "workerId": "worker123"
    }

    # Второй ответ со статусом FAILED
    failed_response = {
        "status": "FAILED",
        "error": "что-то пошло не так"
    }

    # Моки для первого запроса (runsync)
    first_response = AsyncMock()
    first_response.json = AsyncMock(return_value=in_progress_response)
    first_response.raise_for_status = MagicMock()

    first_post_context = AsyncMock()
    first_post_context.__aenter__ = AsyncMock(return_value=first_response)
    first_post_context.__aexit__ = AsyncMock(return_value=None)

    # Моки для второго запроса (status)
    second_response = AsyncMock()
    second_response.json = AsyncMock(return_value=failed_response)

    second_get_context = AsyncMock()
    second_get_context.__aenter__ = AsyncMock(return_value=second_response)
    second_get_context.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.post = MagicMock(return_value=first_post_context)
    session.get = MagicMock(return_value=second_get_context)

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=session)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=client_cm), \
            patch("asyncio.sleep", AsyncMock()):
        result = await llm.run_llm(1, "промпт")
        assert "Апельсин" in result  # Проверяем кодовое слово в сообщении об ошибке
        assert session.get.call_count > 0


@pytest.mark.asyncio
async def test_run_llm_in_progress_completed():
    """Тестирует сценарий, когда RunPod сначала возвращает IN_PROGRESS, а затем COMPLETED"""
    # Первый ответ со статусом IN_PROGRESS
    in_progress_response = {
        "delayTime": 46842,
        "id": "sync-task-id",
        "status": "IN_PROGRESS",
        "workerId": "worker123"
    }

    # Второй ответ со статусом COMPLETED
    completed_response = {
        "status": "COMPLETED",
        "output": {"text": "успешный ответ"}
    }

    # Моки для первого запроса (runsync)
    first_response = AsyncMock()
    first_response.json = AsyncMock(return_value=in_progress_response)
    first_response.raise_for_status = MagicMock()

    first_post_context = AsyncMock()
    first_post_context.__aenter__ = AsyncMock(return_value=first_response)
    first_post_context.__aexit__ = AsyncMock(return_value=None)

    # Моки для второго запроса (status)
    second_response = AsyncMock()
    second_response.json = AsyncMock(return_value=completed_response)

    second_get_context = AsyncMock()
    second_get_context.__aenter__ = AsyncMock(return_value=second_response)
    second_get_context.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.post = MagicMock(return_value=first_post_context)
    session.get = MagicMock(return_value=second_get_context)

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=session)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=client_cm), \
            patch("asyncio.sleep", AsyncMock()):  # Мокаем sleep чтобы тест не ждал
        result = await llm.run_llm(1, "промпт")
        assert result == "успешный ответ"
        assert session.get.call_count > 0  # Проверяем, что был запрос статуса


@pytest.mark.asyncio
async def test_run_llm_success():
    payload = {
        "output": {"text": "ответ"}
    }

    # Корректно настраиваем моки
    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value=payload)
    # Используем MagicMock вместо AsyncMock для синхронного метода
    mock_response.raise_for_status = MagicMock()

    post_context = AsyncMock()
    post_context.__aenter__ = AsyncMock(return_value=mock_response)
    post_context.__aexit__ = AsyncMock(return_value=None)

    session = AsyncMock()
    session.post = MagicMock(return_value=post_context)

    client_cm = AsyncMock()
    client_cm.__aenter__ = AsyncMock(return_value=session)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=client_cm):
        result = await llm.run_llm(1, "промпт")
        assert result == "ответ"


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
async def test_build_messages(monkeypatch):
    monkeypatch.setattr(llm, "get_current_messages", AsyncMock(return_value=[
        {"message": "Hello", "role": "user"},
        {"message": "Hi there", "role": "assistant"}
    ]))
    monkeypatch.setattr(llm, "truncate_history", AsyncMock(side_effect=lambda msgs, max_tokens: msgs))
    monkeypatch.setattr(llm, "find_similar", AsyncMock(return_value=["Арсен: Similar message"]))

    messages = await llm.build_messages(123, "test query")

    assert len(messages) == 3  # системное сообщение + 2 сообщения из истории
    assert messages[0]["role"] == "system"
    assert "Similar message" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there"
