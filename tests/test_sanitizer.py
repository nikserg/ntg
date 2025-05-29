import pytest

import sanitizer


@pytest.mark.parametrize("exclude_words,text,expected", [
    (["кот"], "Это кот и котик.", "Это и котик."),
    (["кот", "собака"], "Кот и собака гуляют. Котик не собака.", "и гуляют. Котик не ."),
    (["test"], "test testing tested", "testing tested"),
    (["word"], "word-word word", "-"),
    (["a"], "a ab ba a", "ab ba"),
])
def test_exclude_words_from_input(monkeypatch, exclude_words, text, expected):
    assert sanitizer.exclude_words_from_input(text, exclude_words) == expected


def test_remove_newlines():
    assert sanitizer.remove_newlines("Привет\nмир") == "Привет мир"
    assert sanitizer.remove_newlines("Многострочный\nтекст\nс переносами") == "Многострочный текст с переносами"
    assert sanitizer.remove_newlines("Текст без переносов") == "Текст без переносов"
    assert sanitizer.remove_newlines("\n\n\n") == "   "
