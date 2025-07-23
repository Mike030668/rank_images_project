# src/rank_images/data_processing.py
"""
Функции для обработки и подготовки данных.

Этот модуль содержит вспомогательные функции для:
- Разбиения текста на фрагменты (chunks).
- Нормализации числовых данных (Z-score).
- Создания структуры данных (DataFrame) на основе изображений и промптов.
"""
import json
import logging
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import pandas as pd
from PIL import Image

# Импорт конфигурационной константы
from .config import MAX_SIG_TOK

# Настройка логгирования
logger = logging.getLogger(__name__)

def _chunks(text: str, max_tok: Optional[int]) -> List[str]:
    """
    Разбивает текст на фрагменты ограниченной длины (в словах).

    Используется для обработки длинных текстовых запросов, которые могут
    превышать лимит токенов модели (например, SigLIP).

    Args:
        text (str): Исходный текст для разбиения.
        max_tok (int | None): Максимальное количество слов в одном фрагменте.
                              Если None или <= 0, возвращается список из одного
                              элемента с исходным текстом.

    Returns:
        List[str]: Список фрагментов текста.
    """
    # Если max_tok не задан или <= 0, возвращаем исходный текст как один фрагмент
    if max_tok is None or max_tok <= 0:
        return [text]

    # Разбиваем текст на слова
    words = text.split()

    # Создаем фрагменты по max_tok слов
    chunks = [
        " ".join(words[i : i + max_tok]) for i in range(0, len(words), max_tok)
    ]

    return chunks

def _z(a: np.ndarray) -> np.ndarray:
    """
    Нормализует массив по Z-оценке (стандартная нормализация).

    Вычитает среднее значение массива и делит на стандартное отклонение.
    Если стандартное отклонение равно 0, возвращает массив нулей той же формы.

    Args:
        a (np.ndarray): Входной одномерный массив NumPy.

    Returns:
        np.ndarray: Нормализованный массив той же формы.
    """
    std_dev = a.std()
    if std_dev > 0:
        # Нормализация: (x - mean) / std
        return (a - a.mean()) / std_dev
    else:
        # Если std == 0, все значения одинаковы, возвращаем массив нулей
        return np.zeros_like(a)

def build_dataframe(
    img_dir: Path, prompts: Union[str, dict, pd.DataFrame, None]
) -> pd.DataFrame:
    """
    Создаёт DataFrame с информацией об изображениях и связанных промптах.

    Поддерживает несколько форматов входных данных для промптов:
    - None: Только имена файлов изображений.
    - str (путь к .json): JSON-файл с ключами 'prompt', 'prompt2', 'negative', 'negative2'.
    - dict: Словарь с ключами 'prompt', 'prompt2', 'negative', 'negative2'.
    - str (произвольный текст): Используется как 'prompt', остальные поля пустые.
    - pd.DataFrame: Готовый DataFrame.

    Args:
        img_dir (Path): Путь к директории с изображениями.
        prompts (str | dict | pd.DataFrame | None): Источник промптов.

    Returns:
        pd.DataFrame: DataFrame с колонками ['image', 'prompt', 'prompt2',
                      'negative', 'negative2']. Если `prompts` None, то только 'image'.

    Raises:
        ValueError: Если в dict/json отсутствуют обязательные ключи.
        TypeError: Если тип `prompts` не поддерживается или значения в dict/json
                   не являются строками.
        FileNotFoundError: Если указанный файл .json не найден.
    """
    # Получаем список изображений в директории
    # Поддерживаемые расширения файлов
    supported_extensions = {".png", ".jpg", ".jpeg"}
    image_files = [
        f.name for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    logger.debug(f"Найдено {len(image_files)} изображений в {img_dir}")

    # --- Обработка различных типов `prompts` ---

    # 1. Если prompts=None, создаём DataFrame только с именами файлов
    if prompts is None:
        logger.info("Промпты не предоставлены. Создаю DataFrame только с именами изображений.")
        return pd.DataFrame({"image": image_files})

    # 2. Если prompts уже DataFrame, возвращаем его как есть
    if isinstance(prompts, pd.DataFrame):
        logger.info("Промпты предоставлены в виде DataFrame.")
        return prompts

    # 3. Если prompts - путь к .json файлу
    if isinstance(prompts, str) and Path(prompts).suffix.lower() == ".json":
        logger.info(f"Загружаю промпты из JSON файла: {prompts}")
        with open(prompts, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)
        # Рекурсивный вызов с загруженным dict
        return build_dataframe(img_dir, prompts_data)

    # 4. Если prompts - словарь
    if isinstance(prompts, dict):
        logger.info("Промпты предоставлены в виде словаря.")
        required_keys = ["prompt", "prompt2", "negative", "negative2"]

        # Проверка наличия всех обязательных ключей
        missing_keys = [k for k in required_keys if k not in prompts]
        if missing_keys:
            raise ValueError(f"В словаре промптов отсутствуют обязательные ключи: {missing_keys}")

        # Проверка, что все значения являются строками
        non_string_keys = [k for k in required_keys if not isinstance(prompts[k], str)]
        if non_string_keys:
            raise TypeError(f"Значения для ключей {non_string_keys} в словаре промптов должны быть строками.")

        # Создаём DataFrame: для каждого изображения дублируем промпты
        base_data = {k: prompts[k] for k in required_keys}
        df_data = [{**base_data, "image": fn} for fn in image_files]
        return pd.DataFrame(df_data)

    # 5. Если prompts - произвольная строка
    if isinstance(prompts, str):
        logger.info("Промпты предоставлены в виде строки. Использую её как 'prompt'.")
        base_data = dict(prompt=prompts, prompt2="", negative="", negative2="")
        df_data = [{**base_data, "image": fn} for fn in image_files]
        return pd.DataFrame(df_data)

    # 6. Если тип prompts не поддерживается
    raise TypeError(
        "Неподдерживаемый тип для `prompts`. "
        "Ожидается: None, str (путь к .json или текст), dict, pd.DataFrame"
    )

# --- Примечание о расширяемости ---
# Функция `build_dataframe` спроектирована так, чтобы быть гибкой.
# При добавлении новых источников промптов (например, из CSV, базы данных)
# можно добавить соответствующие ветви `if isinstance(prompts, ...)` в начало функции.