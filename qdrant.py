import asyncio
import logging
import uuid

import aiohttp
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, VECTOR_SIZE, EMBEDDER_ENDPOINT, CHARACTER_NAME

# Инициализация клиента Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def init_qdrant_collection():
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if QDRANT_COLLECTION_NAME not in collection_names:
        logging.info(f"Создание коллекции {QDRANT_COLLECTION_NAME}")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            )
        )


def qdrant_delete(chat_id):
    qdrant_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=models.Filter(
            must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=str(chat_id)))]
        )
    )


async def embed_text(text):
    # Обращаемся к отдельному сервису для векторизации, повторяем попытки при ошибках
    async with aiohttp.ClientSession() as session:
        retries = 3
        for attempt in range(retries):
            try:
                async with session.post(EMBEDDER_ENDPOINT, json={"text": text}, timeout=30) as response:
                    response.raise_for_status()
                    data = await response.json()
                    embeddings = data.get("embeddings", [])
                    return embeddings
            except Exception as e:
                logging.error(f"Ошибка при векторизации текста: {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
                return None
        return None


async def find_similar(text, chat_id, current_context=None, top_k=3):
    # Генерируем эмбеддинг для поиска
    query_vector = await embed_text(text)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        find_similar_sync,
        chat_id, query_vector, current_context, top_k
    )


def find_similar_sync(chat_id, query_vector, current_context=None, top_k=3):
    """Находит похожие сообщения в Qdrant, исключая те, что уже есть в контексте"""
    if current_context is None:
        current_context = []

    try:
        # Ищем похожие сообщения в Qdrant только для данного chat_id
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="chat_id", match=models.MatchValue(value=str(chat_id)))]
            ),
            limit=top_k * 3  # Запрашиваем больше результатов для фильтрации
        )

        # Фильтруем результаты, исключая те, что уже в контексте
        similar_msgs = []
        for result in search_result:
            candidate = result.payload["text"]
            role_name = "Собеседник" if result.payload["role"] == "user" else CHARACTER_NAME
            if candidate not in current_context:
                similar_msgs.append(f"{role_name}: {candidate}")
                if len(similar_msgs) >= top_k:
                    break

        return similar_msgs
    except Exception as e:
        logging.error(f"Ошибка при поиске в Qdrant: {e}")
        return []


async def save_message_to_qdrant(chat_id, message_text, role):
    """Сохраняет сообщение в Qdrant"""
    try:
        # Генерируем уникальный ID для сообщения (UUID)
        message_id = str(uuid.uuid4())
        emb = await embed_text(message_text)
        # Сохраняем сообщение и его вектор
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=message_id,
                    vector=emb,
                    payload={"chat_id": str(chat_id), "text": message_text, "role": role}
                )
            ]
        )
        return True
    except Exception as e:
        logging.error(f"Ошибка при сохранении в Qdrant: {e}")
        return False
