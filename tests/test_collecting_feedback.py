import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Add the parent directory to sys.path to allow importing modules from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import collecting_feedback
from config import ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK, ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK


@pytest.fixture(autouse=True)
def reset_collecting_feedback():
    """Reset the collecting_feedback dictionary before each test."""
    collecting_feedback.collecting_feedback = {}


@pytest.mark.asyncio
async def test_collect_feedback_empty():
    """Test that an empty feedback is rejected."""
    response, success = await collecting_feedback.collect_feedback(123, "  ")
    assert success is False
    assert "Отзыв не может быть пустым" in response


@pytest.mark.asyncio
async def test_collect_feedback_too_long():
    """Test that a feedback that is too long is rejected."""
    long_feedback = "a" * 2001
    response, success = await collecting_feedback.collect_feedback(123, long_feedback)
    assert success is False
    assert "Отзыв слишком длинный" in response


@pytest.mark.asyncio
async def test_collect_feedback_too_short():
    """Test that a feedback that is too short is rejected."""
    short_feedback = "abc"
    response, success = await collecting_feedback.collect_feedback(123, short_feedback)
    assert success is False
    assert "Отзыв слишком короткий" in response


@pytest.mark.asyncio
async def test_collect_feedback_repeated_chars():
    """Test that feedback with too many repeated characters is rejected."""
    repeated_chars = "This is a test!!!!!!!!!!!!!!!!"
    response, success = await collecting_feedback.collect_feedback(123, repeated_chars)
    assert success is False
    assert "слишком много повторяющихся символов" in response


@pytest.mark.asyncio
async def test_collect_feedback_valid():
    """Test that a valid feedback is accepted and saved."""
    valid_feedback = "This is a valid feedback with sufficient length."

    with patch('db.execute_query', new_callable=AsyncMock) as mock_execute_query:
        mock_execute_query.return_value = None
        response, success = await collecting_feedback.collect_feedback(123, valid_feedback)

        # Check that db.write_feedback was called with the correct arguments
        mock_execute_query.assert_called_once()

        # Check the response
        assert success is True
        assert "Спасибо за ваш отзыв" in response


@pytest.mark.asyncio
async def test_collect_feedback_db_error():
    """Test handling of database errors during feedback collection."""
    valid_feedback = "This is a valid feedback but there's a database error."

    with patch('db.execute_query', new_callable=AsyncMock) as mock_write_feedback:
        # Simulate a database error
        mock_write_feedback.side_effect = Exception("Database error")

        response, success = await collecting_feedback.collect_feedback(123, valid_feedback)

        # Check that db.write_feedback was called
        mock_write_feedback.assert_called_once()

        # Check the response
        assert success is False
        assert "Не удалось сохранить отзыв" in response


def test_handle_command():
    """Test the handle_command function."""
    chat_id = 456
    response = collecting_feedback.handle_command(chat_id)

    # Check that the chat_id is marked as collecting feedback
    assert collecting_feedback.collecting_feedback.get(chat_id) is True

    # Check the response content
    assert str(ADDITIONAL_MESSAGES_PER_DAY_FOR_FEEDBACK) in response
    assert str(ADDITIONAL_MESSAGES_PER_DAY_FOR_USEFUL_FEEDBACK) in response
    assert "Бот сейчас в бета-версии" in response


@pytest.mark.asyncio
async def test_handle_feedback_success():
    """Test the handle_feedback function with successful feedback collection."""
    chat_id = 789
    feedback_text = "This is a good test feedback."

    # Set the chat as collecting feedback
    collecting_feedback.collecting_feedback[chat_id] = True

    with patch('collecting_feedback.collect_feedback', new_callable=AsyncMock) as mock_collect:
        mock_collect.return_value = ("Спасибо за ваш отзыв!", True)

        response = await collecting_feedback.handle_feedback(chat_id, feedback_text)

        # Check that collect_feedback was called correctly
        mock_collect.assert_called_once_with(chat_id, feedback_text)

        # Check that the chat is no longer marked as collecting feedback
        assert collecting_feedback.collecting_feedback.get(chat_id) is False

        # Check the response
        assert "Спасибо за ваш отзыв!" in response


@pytest.mark.asyncio
async def test_handle_feedback_failure():
    """Test the handle_feedback function with failed feedback collection."""
    chat_id = 789
    feedback_text = "a"  # Too short

    # Set the chat as collecting feedback
    collecting_feedback.collecting_feedback[chat_id] = True

    with patch('collecting_feedback.collect_feedback', new_callable=AsyncMock) as mock_collect:
        mock_collect.return_value = ("Отзыв слишком короткий.", False)

        response = await collecting_feedback.handle_feedback(chat_id, feedback_text)

        # Check that collect_feedback was called correctly
        mock_collect.assert_called_once_with(chat_id, feedback_text)

        # Check that the chat is still marked as collecting feedback
        assert collecting_feedback.collecting_feedback.get(chat_id) is True

        # Check the response
        assert "Отзыв слишком короткий" in response


def test_is_collecting_feedback():
    """Test the is_collecting_feedback function."""
    chat_id = 123

    # Initially, should not be collecting feedback
    assert collecting_feedback.is_collecting_feedback(chat_id) is False

    # Set as collecting feedback
    collecting_feedback.collecting_feedback[chat_id] = True
    assert collecting_feedback.is_collecting_feedback(chat_id) is True

    # Set as not collecting feedback
    collecting_feedback.collecting_feedback[chat_id] = False
    assert collecting_feedback.is_collecting_feedback(chat_id) is False
