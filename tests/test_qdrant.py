import sys
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from aioresponses import aioresponses

# Импортируем qdrant, полностью патчируя его зависимости от Qdrant
with patch('qdrant_client.QdrantClient'), \
        patch('qdrant_client.http.api_client.ApiClient'), \
        patch('httpx._client.Client.send'):
    if 'qdrant' in sys.modules:
        del sys.modules['qdrant']  # Сбрасываем кэш импорта
    import qdrant


@pytest.mark.asyncio
async def test_embed_text(monkeypatch):
    from qdrant import embed_text, EMBEDDER_ENDPOINT

    # Мокаем внешний API эмбеддинга
    with aioresponses() as m:
        m.post(EMBEDDER_ENDPOINT, payload={"embeddings": [1, 2, 3]})

        result = await embed_text("тест")
        assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_find_similar_empty(monkeypatch):
    """Тестируем поиск в пустом Qdrant хранилище"""
    chat_id = 123

    # Создаем мок для qdrant_client.search
    mock_search = MagicMock(return_value=[])
    monkeypatch.setattr(qdrant.qdrant_client, "search", mock_search)
    monkeypatch.setattr(qdrant, "get_character_name", AsyncMock(return_value="Арсен"))
    monkeypatch.setattr(qdrant, "embed_text", AsyncMock(return_value=np.ones(384, dtype="float32")))

    # Тестируем функцию
    result = await qdrant.find_similar("тест", chat_id)
    assert result == []

    # Проверяем вызов с правильными параметрами
    mock_search.assert_called_once()
    args, kwargs = mock_search.call_args
    assert kwargs["collection_name"] == qdrant.QDRANT_COLLECTION_NAME
    assert "query_vector" in kwargs
    assert "query_filter" in kwargs

    # Проверка с непустым контекстом
    await qdrant.find_similar("тест", chat_id, current_context=["сообщение"])
    assert mock_search.call_count == 2


@pytest.mark.asyncio
async def test_find_similar_found(monkeypatch):
    """Тестируем поиск с результатами в Qdrant"""
    chat_id = 456

    # Создаем мок для результатов поиска
    class MockQueryResult:
        def __init__(self, payload):
            self.payload = payload

    # Создаем моковые результаты поиска
    mock_results = [
        MockQueryResult({"text": "msg1", "role": "user"}),
        MockQueryResult({"text": "msg2", "role": "assistant"}),
    ]

    # Мокируем embed_text и qdrant_client.search
    monkeypatch.setattr(qdrant, "embed_text", AsyncMock(return_value=np.ones(384, dtype="float32")))
    monkeypatch.setattr(qdrant, "get_character_name", AsyncMock(return_value="Арсен"))
    monkeypatch.setattr(qdrant.qdrant_client, "search", MagicMock(return_value=mock_results))

    # Тестируем функцию
    result = await qdrant.find_similar("тест", chat_id, top_k=2)
    assert result == ["Собеседник: msg1", "Арсен: msg2"]


@pytest.mark.asyncio
async def test_find_similar_with_context_filter(monkeypatch):
    """Тестируем фильтрацию контекста в результатах Qdrant"""
    chat_id = 789

    # Создаем мок для результатов поиска
    class MockQueryResult:
        def __init__(self, payload):
            self.payload = payload

    # Исправляем мокированные результаты поиска
    mock_results = [
        MockQueryResult({"text": "msg1", "role": "assistant"}),
        MockQueryResult({"text": "msg2", "role": "assistant"}),
        MockQueryResult({"text": "msg3", "role": "assistant"}),
    ]

    # Мокируем embed_text и qdrant_client.search
    monkeypatch.setattr(qdrant, "embed_text", AsyncMock(return_value=np.ones(384, dtype="float32")))
    monkeypatch.setattr(qdrant, "get_character_name", AsyncMock(return_value="Арсен"))
    monkeypatch.setattr(qdrant.qdrant_client, "search", MagicMock(return_value=mock_results))

    # Тестируем функцию без фильтрации контекста
    result = await qdrant.find_similar("тест", chat_id, top_k=3)
    assert result == ["Арсен: msg1", "Арсен: msg2", "Арсен: msg3"]

    # Тестируем функцию с фильтрацией контекста
    result = await qdrant.find_similar("тест", chat_id, current_context=["msg1"], top_k=3)
    assert result == ["Арсен: msg2", "Арсен: msg3"]

    # Тестируем функцию с фильтрацией нескольких сообщений
    result = await qdrant.find_similar("тест", chat_id, current_context=["msg1", "msg3"], top_k=3)
    assert result == ["Арсен: msg2"]


@pytest.mark.asyncio
async def test_save_message_to_qdrant(monkeypatch):
    """Тестируем сохранение сообщения в Qdrant"""
    chat_id = 123
    message_text = "Тестовое сообщение"
    message_vector = np.ones(384, dtype="float32")

    # Мокируем qdrant_client.upsert
    mock_upsert = MagicMock(return_value=None)
    monkeypatch.setattr(qdrant.qdrant_client, "upsert", mock_upsert)

    # Мокируем uuid для предсказуемости
    monkeypatch.setattr(qdrant.uuid, "uuid4", MagicMock(return_value="test-uuid"))

    # Мокируем embed_text, чтобы не делать реальный вызов
    monkeypatch.setattr(qdrant, "embed_text", AsyncMock(return_value=message_vector))

    # Тестируем функцию
    result = await qdrant.save_message_to_qdrant(chat_id, message_text, "user")
    assert result == True

    # Проверяем вызов с правильными параметрами
    mock_upsert.assert_called_once()
    args, kwargs = mock_upsert.call_args
    assert kwargs["collection_name"] == qdrant.QDRANT_COLLECTION_NAME
    assert len(kwargs["points"]) == 1
    point = kwargs["points"][0]
    assert point.id == "test-uuid"
