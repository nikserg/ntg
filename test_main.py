from unittest.mock import patch, AsyncMock, MagicMock

import numpy as np
import pytest

import main


def test_trim_incomplete_sentence():
    assert main.trim_incomplete_sentence("Привет! Как дела?") == "Привет! Как дела?"
    assert main.trim_incomplete_sentence("Это тестовое предложение") == "Это тестовое предложение"
    assert main.trim_incomplete_sentence("Привет. Как дела") == "Привет."
    assert main.trim_incomplete_sentence("Тест... Что дальше") == "Тест..."
    assert main.trim_incomplete_sentence("Без знаков окончания") == "Без знаков окончания"


def test_truncate_history():
    # Мокаем токенизатор
    main.tokenizer = MagicMock()
    main.tokenizer.encode = lambda x: [0] * len(x)

    messages = ["a" * 5, "b" * 10, "c" * 20]
    result = main.truncate_history(messages, max_tokens=100)
    assert result == messages  # Проверяем, что все сообщения помещаются

    # Проверяем обрезку истории
    result = main.truncate_history(messages, max_tokens=31)
    assert result == ["b" * 10, "c" * 20]

    result = main.truncate_history(messages, max_tokens=30)
    assert result == ["b" * 10, "c" * 20]

    result = main.truncate_history(messages, max_tokens=20)
    assert result == ["c" * 20]

    result = main.truncate_history(messages, max_tokens=10)
    assert result == []


def test_embed_text():
    with patch.object(main.embedder, "encode", return_value=[np.arange(384, dtype="float32")]):
        emb = main.embed_text("тест")
        assert isinstance(emb, np.ndarray)
        assert emb.shape[0] == 384


def test_find_similar_empty():
    chat_id = 123
    main.vector_embeddings[chat_id] = []
    main.vector_store[chat_id] = []
    assert main.find_similar("тест", chat_id) == []


def test_find_similar_found():
    chat_id = 456
    emb = np.ones(384, dtype="float32")
    main.vector_embeddings[chat_id] = [emb, emb]
    main.vector_store[chat_id] = ["msg1", "msg2"]
    with patch("main.embed_text", return_value=emb):
        with patch("main.NearestNeighbors") as nn_mock:
            nn = nn_mock.return_value
            nn.kneighbors.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
            nn.fit.return_value = None
            result = main.find_similar("тест", chat_id, top_k=2)
            assert result == ["msg1", "msg2"]


def test_build_prompt():
    memories = ["было так", "ещё вот так"]
    history = ["Ты: привет", "Ника: здравствуй"]
    prompt = main.build_prompt(memories, history)
    assert "было так" in prompt
    assert "Ты: привет" in prompt
    assert "Ника:" in prompt


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
        result = await main.run_llm("промпт")
        assert result == "ответ"


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
        result = await main.run_llm("промпт")
        assert "техническая проблема" in result


def test_handle_message_start(monkeypatch):
    message = MagicMock()
    message.chat.id = 1
    message.text = "/start"
    message.answer = AsyncMock()
    monkeypatch.setitem(main.chat_history, 1, [])
    import asyncio
    asyncio.run(main.handle_message(message))
    message.answer.assert_awaited()


@pytest.mark.asyncio
async def test_handle_message_regular(monkeypatch):
    chat_id = 2
    message = MagicMock()
    message.chat.id = chat_id
    message.text = "Привет"
    message.answer = AsyncMock()
    main.chat_history[chat_id] = []
    main.vector_store[chat_id] = []
    main.vector_embeddings[chat_id] = []
    monkeypatch.setattr(main, "embed_text", lambda x: np.ones(384, dtype="float32"))
    monkeypatch.setattr(main, "run_llm", AsyncMock(return_value="Ответ!"))
    monkeypatch.setattr(main.bot, "send_chat_action", AsyncMock())  # Мокаем отправку действия
    await main.handle_message(message)
    message.answer.assert_awaited_with("Ответ!")


def test_on_startup_and_shutdown(monkeypatch):
    app = {}
    monkeypatch.setattr(main.bot, "set_webhook", AsyncMock())
    monkeypatch.setattr(main.bot, "delete_webhook", AsyncMock())
    import asyncio
    asyncio.run(main.on_startup(app))
    asyncio.run(main.on_shutdown(app))
