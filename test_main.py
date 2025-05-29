from unittest.mock import patch, AsyncMock, MagicMock

from aiohttp import web
from aioresponses import aioresponses

# Мокируем Qdrant клиент перед импортом main
with patch('qdrant_client.QdrantClient'), patch('qdrant_client.http.api_client.ApiClient'):
    pass

import pytest
import main


@pytest.fixture(autouse=True)
def mock_db_connection(monkeypatch):
    class MockCursor:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def execute(self, query, *args):
            pass

        async def fetchall(self):
            return []

        async def fetchone(self):
            return None

    class MockConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def cursor(self, *args, **kwargs):
            return MockCursor(*args, **kwargs)

    async def mock_get_db_connection():
        return MockConnection()

    # Убедитесь, что get_db_connection возвращает mock_get_db_connection
    monkeypatch.setattr(main, "get_db_connection", mock_get_db_connection)

def test_trim_incomplete_sentence():
    assert main.trim_incomplete_sentence("Привет! Как дела?") == "Привет! Как дела?"
    assert main.trim_incomplete_sentence("Это тестовое предложение") == "Это тестовое предложение"
    assert main.trim_incomplete_sentence("Привет. Как дела") == "Привет."
    assert main.trim_incomplete_sentence("Тест... Что дальше") == "Тест..."
    assert main.trim_incomplete_sentence("Без знаков окончания") == "Без знаков окончания"
    assert main.trim_incomplete_sentence("Текст заканчивается. на *вот это*") == "Текст заканчивается. на *вот это*"
    assert main.trim_incomplete_sentence("Текст заканчивается. на [вот это]") == "Текст заканчивается. на [вот это]"
    assert main.trim_incomplete_sentence("Текст заканчивается. на вот это[") == "Текст заканчивается."


@pytest.mark.asyncio
async def test_truncate_history(monkeypatch):
    from main import truncate_history, TOKENIZER_ENDPOINT

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
async def test_embed_text(monkeypatch):
    from main import embed_text, EMBEDDER_ENDPOINT

    # Мокаем внешний API эмбеддинга
    with aioresponses() as m:
        m.post(EMBEDDER_ENDPOINT, payload={"embeddings": [1, 2, 3]})

        result = await embed_text("тест")
        assert result == [1, 2, 3]

def test_remove_newlines():
    assert main.remove_newlines("Привет\nмир") == "Привет мир"
    assert main.remove_newlines("Многострочный\nтекст\nс переносами") == "Многострочный текст с переносами"
    assert main.remove_newlines("Текст без переносов") == "Текст без переносов"
    assert main.remove_newlines("\n\n\n") == "   "

@pytest.mark.asyncio
async def test_handle_internal_request_start(monkeypatch):
    # Мокаем все функции, которые вызываются внутри
    monkeypatch.setattr(main, "get_or_create_user", AsyncMock(return_value="invite-code"))
    monkeypatch.setattr(main, "reset_history", AsyncMock())
    monkeypatch.setattr(main, "save_message", AsyncMock())
    request = MagicMock()
    request.json = AsyncMock(return_value={"text": "/start", "chat_id": 1})
    resp = await main.handle_internal_request(request)
    assert isinstance(resp, web.Response)
    assert "Привет" in resp.text

@pytest.mark.asyncio
async def test_handle_internal_request_limit(monkeypatch):
    monkeypatch.setattr(main, "count_daily_messages", AsyncMock(return_value=5))
    monkeypatch.setattr(main, "get_invited_users_count", AsyncMock(return_value=2))
    monkeypatch.setattr(main, "get_user_daily_limit", AsyncMock(return_value=250))
    request = MagicMock()
    request.json = AsyncMock(return_value={"text": "/limit", "chat_id": 1})
    resp = await main.handle_internal_request(request)
    assert "Ваш лимит сообщений" in resp.text


@pytest.mark.asyncio
async def test_handle_internal_request_regular(monkeypatch):
    monkeypatch.setattr(main, "count_daily_messages", AsyncMock(return_value=0))
    monkeypatch.setattr(main, "get_user_daily_limit", AsyncMock(return_value=100))
    monkeypatch.setattr(main, "get_current_messages", AsyncMock(return_value=[]))
    monkeypatch.setattr(main, "save_message", AsyncMock())
    monkeypatch.setattr(main, "exclude_words_from_input", lambda *a, **kw: a[0])
    monkeypatch.setattr(main, "remove_newlines", lambda *a, **kw: a[0])
    monkeypatch.setattr(main, "save_message_to_qdrant", AsyncMock())
    monkeypatch.setattr(main, "trim_incomplete_sentence", lambda *a, **kw: a[0])
    monkeypatch.setattr(main, "build_messages", AsyncMock(return_value=[{"role": "user", "content": "test"}]))
    monkeypatch.setattr(main, "run_llm", AsyncMock(return_value="Ответ!"))
    request = MagicMock()
    request.json = AsyncMock(return_value={"text": "Привет", "chat_id": 1})
    resp = await main.handle_internal_request(request)
    assert resp.text == "Ответ!"


@pytest.mark.asyncio
async def test_collect_feedback(monkeypatch):
    # Мокаем курсор и соединение
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)
    mock_cursor.execute = AsyncMock()

    mock_connection = AsyncMock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=None)
    mock_connection.cursor = MagicMock(return_value=mock_cursor)

    monkeypatch.setattr(main, "get_db_connection", AsyncMock(return_value=mock_connection))
    result = await main.collect_feedback(1, "Спасибо!")
    assert "Отзыв слишком короткий" in result[0]
    result = await main.collect_feedback(1,
                                         "Отлично! Все работает. Спасибо! Теперь я это классный чел и могу делать все что угодно!")
    assert "Спасибо за ваш отзыв" in result[0]
    result = await main.collect_feedback(1, "Это печка!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    assert "Похоже, в вашем отзыве слишком много повторяющихся символов" in result[0]

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

def test_clean_llm_response():
    response = "Это тестовый ответ. С уважением, Арсен"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ. С уважением, Арсен"

    response = "Это тестовый ответ *** С уважением, Арсен"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ С уважением, Арсен"

    response = "Арсен: Это тестовый ответ"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "Это тестовый ответ"

    response = "Ответ &amp; ответ"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "Ответ & ответ"

    response = "Ответ &amp;#34; ответ"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == 'Ответ " ответ'

    response = "Ответ &#34; ответ"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == 'Ответ " ответ'

    response = "ой.&lt;/s&gt; *"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "ой. *"

    response = "Привет [ой] <ой>."
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "Привет ой ой."

    response = "Привет! Я тебя не заметил.\n\nАрсен: Привет, Вася! Как дела?"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "Привет! Я тебя не заметил."

    response = (
        "*улыбается* Ой. "
        "Просто *да*\n\n"
        "Незнакомец: ...\n\n"
        "Арсен: *кивает* "
        "*...*"
    )
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "*улыбается* Ой. Просто *да*"

    response = (
        "*улыбается* Привет! **ОБЪЯСНЕНИЕ:**\n"
        "- 123\n"
        "- 456"
    )
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "*улыбается* Привет!"

    response = (
        "*улыбается* Привет! *ОБЪЯСНЕНИЕ:*\n"
        "- 123\n"
        "- 456"
    )
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "*улыбается* Привет!"

    response = "*привет* Привет (Нет) Да (Сценарий продолжается с"
    cleaned_response = main.clean_llm_response(response)
    assert cleaned_response == "*привет* Привет (Нет) Да"


@pytest.mark.parametrize("exclude_words,text,expected", [
    (["кот"], "Это кот и котик.", "Это и котик."),
    (["кот", "собака"], "Кот и собака гуляют. Котик не собака.", "и гуляют. Котик не ."),
    (["test"], "test testing tested", "testing tested"),
    (["word"], "word-word word", "-"),
    (["a"], "a ab ba a", "ab ba"),
])
def test_exclude_words_from_input(monkeypatch, exclude_words, text, expected):
    assert main.exclude_words_from_input(text, exclude_words) == expected

@pytest.mark.asyncio
async def test_get_or_create_user(monkeypatch):
    # Тест создания нового пользователя
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=None)

    mock_connection = AsyncMock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock()
    mock_connection.cursor = MagicMock(return_value=mock_cursor)

    monkeypatch.setattr(main, "get_db_connection", AsyncMock(return_value=mock_connection))
    monkeypatch.setattr(main.uuid, "uuid4", MagicMock(return_value="test-invite-code"))

    result = await main.get_or_create_user(123)
    assert result == "test-invite-code"
    mock_cursor.execute.assert_any_call(mock_cursor.execute.call_args_list[1].args[0], (123, "test-invite-code", None))

    # Тест получения существующего пользователя
    mock_cursor.fetchone = AsyncMock(return_value=("existing-code",))
    result = await main.get_or_create_user(456)
    assert result == "existing-code"


@pytest.mark.asyncio
async def test_build_messages(monkeypatch):
    monkeypatch.setattr(main, "get_current_messages", AsyncMock(return_value=[
        {"message": "Hello", "role": "user"},
        {"message": "Hi there", "role": "assistant"}
    ]))
    monkeypatch.setattr(main, "truncate_history", AsyncMock(side_effect=lambda msgs, max_tokens: msgs))
    monkeypatch.setattr(main, "find_similar", AsyncMock(return_value=["Арсен: Similar message"]))

    messages = await main.build_messages(123, "test query")

    assert len(messages) == 3  # системное сообщение + 2 сообщения из истории
    assert messages[0]["role"] == "system"
    assert "Similar message" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there"

@pytest.mark.asyncio
async def test_apply_migrations(monkeypatch):
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)

    mock_connection = AsyncMock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=None)
    mock_connection.cursor = MagicMock(return_value=mock_cursor)

    monkeypatch.setattr(main, "get_db_connection", AsyncMock(return_value=mock_connection))

    await main.apply_migrations()

    # Проверяем, что были выполнены 3 SQL-запроса для создания таблиц
    assert mock_cursor.execute.call_count >= 3


@pytest.mark.asyncio
async def test_reset_history(monkeypatch):
    chat_id = 123

    # Мок для подключения к БД
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)

    mock_connection = AsyncMock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=None)
    mock_connection.cursor = MagicMock(return_value=mock_cursor)

    # Мок для Qdrant
    monkeypatch.setattr(main.qdrant_client, "delete", MagicMock())
    monkeypatch.setattr(main, "get_db_connection", AsyncMock(return_value=mock_connection))

    # Вызываем функцию сброса истории
    await main.reset_history(chat_id)

    # Проверяем вызовы
    mock_cursor.execute.assert_called_once()
    main.qdrant_client.delete.assert_called_once()
