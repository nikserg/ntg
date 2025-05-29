from unittest.mock import patch, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from main import handle_internal_request, on_startup, app


@pytest.fixture
def app_fixture():
    """Create a test application."""
    return app


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.collecting_feedback.handle_feedback')
async def test_handle_internal_request_collecting_feedback(mock_handle_feedback, mock_is_collecting_feedback):
    """Test handling a request when collecting feedback."""
    mock_is_collecting_feedback.return_value = True
    mock_handle_feedback.return_value = "Thank you for your feedback!"

    request_data = {"text": "This is my feedback", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    # Create a proper async mock for the json method
    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "Thank you for your feedback!"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_handle_feedback.assert_called_once_with("12345", "This is my feedback")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.start.handle_command')
async def test_handle_internal_request_start_command(mock_start_handle_command, mock_is_collecting_feedback):
    """Test handling a /start command."""
    mock_is_collecting_feedback.return_value = False
    mock_start_handle_command.return_value = "Welcome message"

    request_data = {"text": "/start", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "Welcome message"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_start_handle_command.assert_called_once_with("12345", "/start")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.collecting_feedback.handle_command')
async def test_handle_internal_request_feedback_command(mock_feedback_handle_command, mock_is_collecting_feedback):
    """Test handling a /feedback command."""
    mock_is_collecting_feedback.return_value = False
    mock_feedback_handle_command.return_value = "Please provide your feedback"

    request_data = {"text": "/feedback", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "Please provide your feedback"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_feedback_handle_command.assert_called_once_with("12345")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.invite.handle_command')
async def test_handle_internal_request_invite_command(mock_invite_handle_command, mock_is_collecting_feedback):
    """Test handling an /invite command."""
    mock_is_collecting_feedback.return_value = False
    mock_invite_handle_command.return_value = "Here's your invite link"

    request_data = {"text": "/invite", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "Here's your invite link"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_invite_handle_command.assert_called_once_with("12345")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.limits.handle_command')
async def test_handle_internal_request_limit_command(mock_limits_handle_command, mock_is_collecting_feedback):
    """Test handling a /limit command."""
    mock_is_collecting_feedback.return_value = False
    mock_limits_handle_command.return_value = "Your current usage limits"

    request_data = {"text": "/limit", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "Your current usage limits"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_limits_handle_command.assert_called_once_with("12345")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.limits.is_limit_exceeded')
async def test_handle_internal_request_limit_exceeded(mock_is_limit_exceeded, mock_is_collecting_feedback):
    """Test handling a request when the user has exceeded their message limit."""
    mock_is_collecting_feedback.return_value = False
    mock_is_limit_exceeded.return_value = (True, "You've exceeded your message limit")

    request_data = {"text": "Hello bot", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "You've exceeded your message limit"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_is_limit_exceeded.assert_called_once_with("12345")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
@patch('main.limits.is_limit_exceeded')
@patch('main.get_current_messages')
@patch('main.start.add_first_messages_to_db')
@patch('main.remove_newlines')
@patch('main.exclude_words_from_input')
@patch('main.save_message')
@patch('main.qdrant.save_message_to_qdrant')
@patch('main.llm.run_llm')
async def test_handle_internal_request_normal_message(
        mock_run_llm, mock_save_to_qdrant, mock_save_message,
        mock_exclude_words, mock_remove_newlines, mock_add_first_messages,
        mock_get_messages, mock_is_limit_exceeded, mock_is_collecting_feedback
):
    """Test handling a normal message with successful LLM response."""
    mock_is_collecting_feedback.return_value = False
    mock_is_limit_exceeded.return_value = (False, "")
    mock_get_messages.return_value = None
    mock_add_first_messages.return_value = None
    mock_remove_newlines.return_value = "Hello bot"
    mock_exclude_words.return_value = "Hello bot"
    mock_save_message.return_value = None
    mock_save_to_qdrant.return_value = None
    mock_run_llm.return_value = "Hello human!"

    request_data = {"text": "Hello bot", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 200
    assert response.text == "Hello human!"
    mock_is_collecting_feedback.assert_called_once_with("12345")
    mock_is_limit_exceeded.assert_called_once_with("12345")
    mock_get_messages.assert_called_once_with("12345")
    mock_add_first_messages.assert_called_once_with("12345")

    # Check that remove_newlines was called twice: first with user input, then with bot response
    assert mock_remove_newlines.call_count == 2
    # Check the first call was with the user message
    mock_remove_newlines.assert_any_call("Hello bot")
    # Check the second call was with the bot response
    mock_remove_newlines.assert_any_call("Hello human!")

    mock_exclude_words.assert_called_once()
    assert mock_save_message.call_count == 2  # Save user message and assistant response
    assert mock_save_to_qdrant.call_count == 2  # Save user message and assistant response to Qdrant
    mock_run_llm.assert_called_once_with("12345", "Hello bot")


@pytest.mark.asyncio
@patch('main.collecting_feedback.is_collecting_feedback')
async def test_handle_internal_request_exception(mock_is_collecting_feedback):
    """Test handling exceptions in the request handler."""
    mock_is_collecting_feedback.side_effect = Exception("Test exception")

    request_data = {"text": "Hello bot", "chat_id": "12345"}
    request = make_mocked_request('POST', '/internal')

    async def mock_json():
        return request_data

    request.json = mock_json

    response = await handle_internal_request(request)

    assert response.status == 500
    assert "Ошибка: Test exception" in response.text


@pytest.mark.asyncio
@patch('main.qdrant.init_qdrant_collection')
@patch('main.migrations.apply_migrations')
async def test_on_startup(mock_apply_migrations, mock_init_qdrant):
    """Test the application startup function."""
    mock_app = MagicMock()
    await on_startup(mock_app)

    # Verify that initializations were called
    assert mock_init_qdrant.call_count == 1
    assert mock_apply_migrations.call_count == 1


def test_app_routes():
    """Test that the application has the expected routes."""
    route_found = False
    for route in app.router.routes():
        if isinstance(route, web.ResourceRoute) and route.method == "POST" and str(
                route.resource.canonical) == "/internal":
            route_found = True
            break

    assert route_found, "Expected POST /internal route not found"

    # Check that the startup handler is registered
    assert len(app.on_startup) > 0, "No startup handlers registered"
