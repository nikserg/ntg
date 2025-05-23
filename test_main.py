import asyncio
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
    # Проверка с непустым контекстом
    assert main.find_similar("тест", chat_id, current_context=["сообщение"]) == []


def test_remove_newlines():
    assert main.remove_newlines("Привет\nмир") == "Привет мир"
    assert main.remove_newlines("Многострочный\nтекст\nс переносами") == "Многострочный текст с переносами"
    assert main.remove_newlines("Текст без переносов") == "Текст без переносов"
    assert main.remove_newlines("\n\n\n") == "   "


@pytest.mark.asyncio
async def test_typing_action():
    # Мокируем бота
    mock_bot = MagicMock()
    mock_bot.send_chat_action = AsyncMock()

    # Сохраняем оригинальный бот и заменяем его моком
    original_bot = main.bot
    main.bot = mock_bot

    try:
        # Проверяем нормальное выполнение в контексте
        async with main.typing_action(123):
            await asyncio.sleep(0.2)  # Даем время для выполнения keep_typing

        # Проверяем, что метод вызывался с правильными параметрами
        mock_bot.send_chat_action.assert_awaited_with(123, 'typing')
        assert mock_bot.send_chat_action.await_count >= 1

        # Сбрасываем счетчик вызовов
        mock_bot.send_chat_action.reset_mock()

        # Проверяем обработку исключений внутри контекста
        with pytest.raises(ValueError):
            async with main.typing_action(456):
                await asyncio.sleep(0.1)
                raise ValueError("Тестовое исключение")

        # Статус должен вызываться даже при исключении
        mock_bot.send_chat_action.assert_awaited_with(456, 'typing')
    finally:
        # Восстанавливаем оригинальный бот
        main.bot = original_bot


@pytest.mark.asyncio
async def test_handle_message_with_typing_status(monkeypatch):
    """Проверяет, что статус 'печатает' поддерживается во время обработки сообщения"""
    chat_id = 888
    message = MagicMock()
    message.chat.id = chat_id
    message.text = "Тест на долгую обработку"
    message.answer = AsyncMock()

    # Настраиваем хранилища
    main.chat_history[chat_id] = []
    main.vector_store[chat_id] = []
    main.vector_embeddings[chat_id] = []

    # Мокаем функции
    monkeypatch.setattr(main, "embed_text", lambda x: np.ones(384, dtype="float32"))
    monkeypatch.setattr(main.bot, "send_chat_action", AsyncMock())

    # Создаем искусственную задержку в run_llm для проверки
    async def slow_llm_response(*args, **kwargs):
        await asyncio.sleep(0.5)  # Имитируем долгое выполнение запроса
        return "Ответ после паузы"

    monkeypatch.setattr(main, "run_llm", slow_llm_response)

    # Обрабатываем сообщение
    await main.handle_message(message)

    # Проверяем, что бот поддерживал статус "печатает"
    main.bot.send_chat_action.assert_awaited_with(chat_id, 'typing')
    assert main.bot.send_chat_action.await_count >= 1

    # Проверяем, что ответ был отправлен
    message.answer.assert_awaited_once_with("Ответ после паузы")


@pytest.mark.asyncio
async def test_handle_first_message_not_start(monkeypatch):
    """Тестирует обработку первого сообщения (не команда /start)"""
    chat_id = 999
    message = MagicMock()
    message.chat.id = chat_id
    message.text = "Привет! Я новый пользователь"
    message.answer = AsyncMock()

    # Удаляем историю, если она уже существует
    if chat_id in main.chat_history:
        del main.chat_history[chat_id]

    # Мокаем необходимые функции
    monkeypatch.setattr(main, "embed_text", lambda x: np.ones(384, dtype="float32"))
    monkeypatch.setattr(main, "run_llm", AsyncMock(return_value="Привет, новый пользователь!"))
    monkeypatch.setattr(main.bot, "send_chat_action", AsyncMock())

    await main.handle_message(message)

    # Проверяем, что история инициализирована
    assert chat_id in main.chat_history
    assert main.chat_history[chat_id][0].startswith("Ника:")

    # Проверяем, что ответ отправлен
    message.answer.assert_awaited_once_with("Привет, новый пользователь!")


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


def test_find_similar_with_context_filter():
    chat_id = 789
    emb = np.ones(384, dtype="float32")
    main.vector_embeddings[chat_id] = [emb, emb, emb]
    main.vector_store[chat_id] = ["msg1", "msg2", "msg3"]
    with patch("main.embed_text", return_value=emb):
        with patch("main.NearestNeighbors") as nn_mock:
            nn = nn_mock.return_value
            nn.kneighbors.return_value = (np.array([[0.1, 0.2, 0.3]]), np.array([[0, 1, 2]]))
            nn.fit.return_value = None

            # Без фильтрации контекста
            result = main.find_similar("тест", chat_id, top_k=3)
            assert result == ["msg1", "msg2", "msg3"]

            # С фильтрацией контекста
            result = main.find_similar("тест", chat_id, current_context=["msg1"], top_k=3)
            assert result == ["msg2", "msg3"]

            # С фильтрацией нескольких сообщений
            result = main.find_similar("тест", chat_id, current_context=["msg1", "msg3"], top_k=3)
            assert result == ["msg2"]


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
        result = await main.run_llm("промпт")
        assert result == "успешный ответ"
        assert session.get.call_count > 0  # Проверяем, что был запрос статуса


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
        result = await main.run_llm("промпт")
        assert "Апельсин" in result  # Проверяем кодовое слово в сообщении об ошибке
        assert session.get.call_count > 0


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
