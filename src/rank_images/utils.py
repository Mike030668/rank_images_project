# src/rank_images/utils.py
"""
Утилиты общего назначения для проекта rank_images.
"""
import logging
from typing import List, Dict, Any
import numpy as np
from PIL import Image

from .data_processing import _z # Импортируем существующую функцию нормализации

logger = logging.getLogger(__name__)

def normalize_metrics(
    results: List[Dict[str, Any]], 
    metric_names: List[str]
) -> Dict[str, np.ndarray]:
    """
    Нормализует указанные метрики в списке результатов по Z-оценке.

    Args:
        results (List[Dict[str, Any]]): Список словарей с результатами для каждого изображения.
                                        Ожидается, что каждый словарь имеет ключи из `metric_names`.
        metric_names (List[str]): Список имен метрик для нормализации.

    Returns:
        Dict[str, np.ndarray]: Словарь, где ключ - имя метрики + "_norm", 
                               значение - массив нормализованных значений.
                               Например, {"sig_norm": np.array([...]), ...}
    """
    normalized_results = {}
    for metric_name in metric_names:
        try:
            # Извлекаем массив значений для текущей метрики
            metric_values = np.array([r[metric_name] for r in results])
            # Нормализуем
            normalized_values = _z(metric_values)
            # Сохраняем с суффиксом "_norm"
            normalized_results[f"{metric_name}_norm"] = normalized_values
        except KeyError:
            logger.warning(f"Метрика '{metric_name}' не найдена в результатах для нормализации.")
        except Exception as e:
            logger.error(f"Ошибка при нормализации метрики '{metric_name}': {e}")
            # Можно либо пропустить, либо поднять исключение
            # normalized_results[f"{metric_name}_norm"] = np.zeros(len(results)) # Или другая заглушка
    
    return normalized_results


def calculate_net_score(
    get_score_func, 
    img: Image.Image, 
    positive_prompts: List[str], 
    negative_prompts: List[str],
    **kwargs # Для передачи дополнительных аргументов функции get_score_func
) -> float:
    """
    Универсальная функция для вычисления "чистого" скоринга:
    Score(позитив) - Score(негатив).

    Args:
        get_score_func (callable): Функция, вычисляющая скор (например, get_siglip_score).
        img (Image.Image): Изображение.
        positive_prompts (List[str]): Список позитивных промптов.
        negative_prompts (List[str]): Список негативных промптов.
        **kwargs: Дополнительные аргументы для get_score_func.

    Returns:
        float: Чистый скор (позитив - негатив). Если список пуст, возвращает 0.0 для этой части.
    """
    pos_score = get_score_func(img, positive_prompts, **kwargs) if positive_prompts else 0.0
    neg_score = get_score_func(img, negative_prompts, **kwargs) if negative_prompts else 0.0
    return pos_score - neg_score
