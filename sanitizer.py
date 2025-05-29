import re


def remove_newlines(text):
    """Удаляет переносы строк из текста, заменяя их на пробелы"""
    return text.replace("\n", " ")


def exclude_words_from_input(text, exclude_words):
    for word in exclude_words:
        # Удаляет слово только если оно встречается как отдельное слово (учитывает границы)
        text = re.sub(rf'\b{re.escape(word)}\b', '', text, flags=re.IGNORECASE)
    # Убираем лишние пробелы после удаления слов
    return re.sub(r'\s{2,}', ' ', text).strip()
