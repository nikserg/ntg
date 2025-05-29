from unittest.mock import AsyncMock, MagicMock

import pytest

import db
import migrations


@pytest.mark.asyncio
async def test_apply_migrations(monkeypatch):
    mock_cursor = AsyncMock()
    mock_cursor.__aenter__ = AsyncMock(return_value=mock_cursor)
    mock_cursor.__aexit__ = AsyncMock(return_value=None)

    mock_connection = AsyncMock()
    mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_connection.__aexit__ = AsyncMock(return_value=None)
    mock_connection.cursor = MagicMock(return_value=mock_cursor)

    monkeypatch.setattr(db, "get_db_connection", AsyncMock(return_value=mock_connection))

    await migrations.apply_migrations()

    # Проверяем, что были выполнены 3 SQL-запроса для создания таблиц
    assert mock_cursor.execute.call_count >= 3
