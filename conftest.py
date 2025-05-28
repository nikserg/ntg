from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# Создаем сложный мок для Qdrant
@pytest.fixture(scope="session", autouse=True)
def mock_qdrant():
    # Мок для коллекций
    mock_collection = MagicMock()
    mock_collection.name = "test_collection"
    mock_collections = MagicMock()
    mock_collections.collections = [mock_collection]

    # Мок для клиента Qdrant
    mock_client = MagicMock()
    mock_client.get_collections.return_value = mock_collections
    mock_client.search.return_value = []
    mock_client.upsert.return_value = None

    # Патчим все необходимые компоненты Qdrant
    with patch('qdrant_client.QdrantClient', return_value=mock_client) as mock_qdrant_cls:
        with patch('qdrant_client.http.api_client.ApiClient') as mock_api_client:
            # Важно - предотвращаем реальные HTTP-запросы
            with patch('httpx._client.Client.send') as mock_send:
                yield mock_client


@pytest.fixture
def mock_db_connection():
    """Мокируем асинхронное подключение к базе данных."""
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_conn.cursor = AsyncMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__aenter__.return_value = mock_cursor
    mock_cursor.__aexit__.return_value = None
    mock_conn.__aenter__.return_value = mock_conn
    return mock_conn
