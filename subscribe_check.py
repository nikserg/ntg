import asyncio
import logging
import time

import aiohttp

from config import SUBSCRIBE_CHECK_ENDPOINT

# In-memory cache for subscription status
# Structure: {chat_id: (is_subscribed, timestamp)}
_subscription_cache = {}
_CACHE_DURATION = 60 * 3


async def check_subscription(chat_id: int, max_retries: int = 3, retry_delay: int = 1) -> bool:
    """
    Checks if chat_id is subscribed to the channel.
    Results are cached for 1 minute to reduce API calls.

    Args:
        chat_id: Telegram chat ID to check
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        bool: True if subscribed, False otherwise
    """
    current_time = time.time()

    # Check if we have a valid cached result
    if chat_id in _subscription_cache:
        is_subscribed, timestamp = _subscription_cache[chat_id]
        if current_time - timestamp < _CACHE_DURATION:
            logging.debug(f"Using cached subscription status for chat_id {chat_id}: {is_subscribed}")
            return is_subscribed

    logging.info(f"Checking subscription status for chat_id: {chat_id}")
    params = {"chat_id": chat_id}

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(SUBSCRIBE_CHECK_ENDPOINT, params=params, timeout=10) as response:
                    if response.status != 200:
                        logging.warning(f"Non-200 response: {response.status} from subscription check API")
                        response.raise_for_status()

                    data = await response.json()
                    is_subscribed = data.get("is_subscribed", False)
                    logging.info(f"Subscription status for chat_id {chat_id}: {is_subscribed}")

                    # Update cache
                    _subscription_cache[chat_id] = (is_subscribed, current_time)
                    return is_subscribed

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.error(f"Error checking subscription (attempt {attempt + 1}/{max_retries}): {e}")

            if attempt == max_retries - 1:
                logging.error(f"Max retries reached checking subscription for chat_id {chat_id}")
                return False

            # Exponential backoff
            await asyncio.sleep(retry_delay * (attempt + 1))

    return False
