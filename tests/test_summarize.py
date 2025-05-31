import os
import sys
from unittest.mock import AsyncMock

import pytest
from aioresponses import aioresponses

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import summarize


@pytest.mark.asyncio
async def test_truncate_history(monkeypatch):
    messages = [
        {"message": "a" * 5, "token_count": 5},
        {"message": "b" * 10, "token_count": 10},
        {"message": "c" * 20, "token_count": 20}
    ]

    with aioresponses() as m:
        # Mock CONTEXT_TOKEN_LIMIT to a specific value
        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 100)
        result = await summarize._truncate_history(messages)
        assert result == messages

        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 31)
        result = await summarize._truncate_history(messages)
        assert result == [{"message": "b" * 10, "token_count": 10}, {"message": "c" * 20, "token_count": 20}]

        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 20)
        result = await summarize._truncate_history(messages)
        assert result == [{"message": "c" * 20, "token_count": 20}]

        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 19)
        result = await summarize._truncate_history(messages)
        assert result == []


@pytest.mark.asyncio
async def test_needs_summarization(monkeypatch):
    messages = [
        {"message": "a" * 5, "token_count": 5},
        {"message": "b" * 10, "token_count": 10},
        {"message": "c" * 20, "token_count": 20}
    ]

    with aioresponses() as m:
        # Mock CONTEXT_TOKEN_LIMIT to a specific value
        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 100)
        result = await summarize.needs_summarization(messages)
        assert not result

        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 31)
        result = await summarize.needs_summarization(messages)
        assert result

        monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 19)
        result = await summarize.needs_summarization(messages)
        assert result


@pytest.mark.asyncio
async def test_mark_messages_as_summarized(monkeypatch):
    messages = [
        {"id": 1, "message": "a" * 5, "role": "user", "token_count": 5},
        {"id": 2, "message": "b" * 10, "role": "assistant", "token_count": 10}
    ]

    # Mock the database operation
    mock_execute_query = AsyncMock()
    monkeypatch.setattr(summarize, "execute_query", mock_execute_query)

    await summarize.mark_messages_as_summarized(messages)

    # Check if execute_query was called with the correct SQL and parameters
    mock_execute_query.assert_called_once_with(
        "UPDATE messages SET summarized = 1 WHERE id IN (%s, %s)",
        [1, 2]
    )


@pytest.mark.asyncio
async def test_count_tokens_in_messages(monkeypatch):
    messages = [
        {"message": "a" * 5, "token_count": 5},
        {"message": "b" * 10, "token_count": 10},
        {"message": "c" * 20, "token_count": 20}
    ]

    total_tokens = await summarize._count_tokens_in_messages(messages)
    assert total_tokens == 35


@pytest.mark.asyncio
async def test_get_summarize_buffer(monkeypatch):
    messages = [
        {"id": 1, "message": "a", "token_count": 5},
        {"id": 2, "message": "b", "token_count": 10},
        {"id": 3, "message": "c", "token_count": 20},
        {"id": 4, "message": "d", "token_count": 15},
        {"id": 5, "message": "e", "token_count": 25}
    ]

    # Mock CONTEXT_TOKEN_LIMIT and SUMMARIZE_BUFFER_PERCENT
    monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 100)
    monkeypatch.setattr(summarize, "SUMMARIZE_BUFFER_PERCENT", 30)

    messages_to_summarize, messages_without_summarize = await summarize.get_summarize_buffer(messages)
    assert messages_to_summarize == [
        {"id": 1, "message": "a", "token_count": 5},
        {"id": 2, "message": "b", "token_count": 10},
        {"id": 3, "message": "c", "token_count": 20}
    ]
    assert messages_without_summarize == [
        {"id": 4, "message": "d", "token_count": 15},
        {"id": 5, "message": "e", "token_count": 25}
    ]

    monkeypatch.setattr(summarize, "CONTEXT_TOKEN_LIMIT", 100)
    monkeypatch.setattr(summarize, "SUMMARIZE_BUFFER_PERCENT", 50)
    messages_to_summarize, messages_without_summarize = await summarize.get_summarize_buffer(messages)
    assert messages_to_summarize == [
        {"id": 1, "message": "a", "token_count": 5},
        {"id": 2, "message": "b", "token_count": 10},
        {"id": 3, "message": "c", "token_count": 20},
        {"id": 4, "message": "d", "token_count": 15}
    ]
    assert messages_without_summarize == [
        {"id": 5, "message": "e", "token_count": 25}
    ]
