# src/rank_images/example_metric.py
"""
Пример шаблона для добавления новой метрики.

Этот файл служит руководством для создания новых функций оценки изображений.
Скопируйте и переименуйте его, затем реализуйте логику новой метрики.
"""
import logging
from typing import List
import torch
from PIL import Image

# Импортируем модуль models целиком для доступа к глобальным переменным моделей
# Это необходимо для правильной работы с моделями, загружаемыми в models.py
from . import models
# Импортируем утилиты, если они нужны
# from .device_utils import _to_gpu, _release # Раскомментируйте при необходимости

logger = logging.getLogger(__name__)

# --- Шаблон новой метрики ---
@torch.inference_mode() # Всегда используем для инференса
def get_new_metric_score(img: Image.Image, prompts: List[str], **kwargs) -> float:
    """
    Краткое описание новой метрики.

    Args:
        img (PIL.Image.Image): Входное изображение.
        prompts (List[str]): Список текстовых промптов.
        **kwargs: Дополнительные аргументы, специфичные для метрики.

    Returns:
        float: Оценка по новой метрике. В случае ошибки или отсутствия модели
               рекомендуется возвращать 0.0 или другое нейтральное значение.
    """
    # 1. Проверка наличия необходимой модели
    # if models.new_model is None or models.new_processor is None:
    #     logger.warning("Модель NewMetric не загружена. Возвращаю 0.0.")
    #     return 0.0

    # 2. Обработка входных данных (изображение, текст)
    # try:
    #     inputs = models.new_processor(images=img, text=prompts, ...)
    # except Exception as e:
    #     logger.error(f"Ошибка при подготовке данных для NewMetric: {e}")
    #     return 0.0

    # 3. Перемещение модели и данных на устройство (если необходимо)
    # model_gpu = _to_gpu(models.new_model) # Если модель должна быть на GPU
    # inputs = {k: v.to(model_gpu.device) for k, v in inputs.items()}

    # 4. Инференс
    # try:
    #     outputs = model_gpu(**inputs)
    # finally:
    #     _release(model_gpu) # Не забываем освободить модель

    # 5. Пост-обработка и вычисление финального балла
    # score = ... # Ваша логика

    # 6. Логирование результата (опционально, но полезно)
    # logger.debug(f"NewMetric скор: {score:.4f}")
    
    # 7. Возврат результата
    # return score

    # --- ВРЕМЕННАЯ ЗАГЛУШКА ---
    logger.info("Функция новой метрики еще не реализована.")
    return 0.0

# --- Конец шаблона ---