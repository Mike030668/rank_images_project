# src/rank_images/ranking.py
"""
Основная логика ранжирования изображений.

Этот модуль содержит функцию `rank_folder`, которая выполняет полный цикл:
1. Загрузка списка изображений и промптов.
2. Вычисление метрик для каждого изображения.
3. Нормализация метрик.
4. Вычисление итогового балла.
5. Сортировка и сохранение результата.

Это центральный модуль для интеграции новых метрик:
- Импортируйте новую функцию метрики.
- Вызовите её в цикле обработки изображений.
- Добавьте её результат в `res`.
- Добавьте её вес в аргументы `rank_folder`.
- Включите её в нормализацию и вычисление `total`.
"""
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# Импорт вспомогательных функций и компонентов
from .utils import normalize_metrics, calculate_net_score
from .data_processing import build_dataframe, _chunks
from .metrics import (
    get_siglip_score,
    get_florence_score,
    get_iqa,
    get_dino,
    get_blip2_match_score
)
# Импорт конфигурации
from .config import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    # --- НОВОЕ ---
    EPSILON_DEFAULT,
    # ------------
    MAX_SIG_TOK  # Максимальное количество токенов для SigLIP
)

# Настройка логгирования
logger = logging.getLogger(__name__)


def rank_folder(
    img_dir: Path,
    prompts_in: Optional[Union[str, dict, pd.DataFrame]] = None,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    delta: float = DELTA_DEFAULT,
    epsilon: float = EPSILON_DEFAULT, # Вес для BLIP-2
    chunk_size: Optional[int] = None,  # Может быть None
) -> pd.DataFrame:
    """
    Ранжирует изображения в заданной директории на основе текстовых промптов
    и внутреннего качества.

    Алгоритм:
    1. Строит DataFrame с изображениями и промптами.
    2. Для каждого изображения:
       a. Вычисляет метрики SigLIP, Florence, IQA, DINO.
       b. Нормализует значения метрик по Z-оценке.
       c. Вычисляет итоговый балл как взвешенную сумму нормализованных метрик.
    3. Сортирует изображения по убыванию итогового балла.
    4. Сохраняет результат в `img_dir/ranking.csv`.

    Args:
        img_dir (Path): Путь к директории с изображениями для ранжирования.
        prompts_in (str | dict | pd.DataFrame | None): Источник промптов.
            - None: Используются только имена файлов.
            - str (путь к .json): JSON-файл с промптами.
            - dict: Словарь с промптами.
            - str (произвольный текст): Используется как 'prompt'.
            - pd.DataFrame: Готовый DataFrame с промптами.
        alpha (float): Вес метрики SigLIP (схожесть изображения и текста).
                       По умолчанию 0.6.
        beta (float): Вес метрики Florence-2 (поиск объектов по запросу).
                      По умолчанию 0.4.
        gamma (float): Вес метрики CLIP-IQA (общее качество изображения).
                       По умолчанию 0.2.
        delta (float): Вес метрики DINOv2 (внутренние признаки изображения).
                       По умолчанию 0.1.
        chunk_size (int | None): Максимальное количество токенов в одном
                                 фрагменте текста для SigLIP. Если None,
                                 используется значение по умолчанию из config.

    Returns:
        pd.DataFrame: DataFrame с результатами ранжирования, отсортированный
                      по убыванию итогового балла. Также сохраняется в
                      `img_dir/ranking.csv`.
                      
    Raises:
        RuntimeError: Если не удалось обработать ни одно изображение.
        FileNotFoundError: Если указанный файл изображения не найден.
        Exception: При ошибках в процессе вычисления метрик.
    """
    logger.info(f"Начинаю ранжирование изображений в папке: {img_dir}")

    # --- 1. Подготовка данных ---
    # Создаем DataFrame с информацией об изображениях и промптах
    df_prompts = build_dataframe(img_dir, prompts_in)
    
    # Убеждаемся, что все необходимые колонки присутствуют
    required_columns = ["prompt", "prompt2", "negative", "negative2"]
    for col in required_columns:
        if col not in df_prompts.columns:
            df_prompts[col] = "" # Заполняем пустыми строками, если отсутствуют

    results = [] # Список для хранения результатов по каждому изображению
    logger.info("Начинаю вычисление метрик для изображений...")

    # --- ЛОКАЛЬНОЕ определение _chunks для ранжирования ---
    # Воссоздаёт логику оригинального скрипта для обработки chunk_size=None
    def _chunks(text: str, max_tok: Optional[int]) -> List[str]:
        """
        Локальная функция разбиения текста на фрагменты.
        
        Если max_tok равно None, используется фиксированное значение MAX_SIG_TOK.
        Это необходимо для корректной работы SigLIP-2, которая имеет ограничение
        на длину входного текста.
        """
        # ВАЖНО: Обработка None для SigLIP (воспроизведение оригинальной логики)
        if max_tok is None:
            max_tok = MAX_SIG_TOK # <-- Подставляем MAX_SIG_TOK если None
        # Обработка <= 0
        if max_tok <= 0:
            return [text]
        # Разбиение на фрагменты
        words = text.split()
        return [" ".join(words[i : i + max_tok]) for i in range(0, len(words), max_tok)]
    # --- Конец локального определения _chunks ---

    # --- Подготовка текстовых запросов ---
    # Функция для создания списка непустых фрагментов текста
    # Использует ЛОКАЛЬНУЮ _chunks
    def _make_chunks(*texts: str) -> List[str]:
        # --- Отладочные принты внутри _make_chunks ---
        #print(f"[DEBUG][_make_chunks] type(_chunks): {type(_chunks)}")
        #print(f"[DEBUG][_make_chunks] _chunks is None: {_chunks is None}")
        if _chunks is None:
            raise RuntimeError("Локальная переменная _chunks равна None внутри _make_chunks!")
        # --- Конец отладочных принтов ---
        
        output_chunks = []
        for text in texts:
            # Разбиваем каждый текст на фрагменты и добавляем непустые
            output_chunks.extend(_chunks(text, chunk_size))
        # Возвращаем список уникальных непустых фрагментов
        return [chunk for chunk in output_chunks if chunk.strip()]


    # --- 2. Цикл по изображениям ---
    # Используем tqdm для отображения прогресса
    for row in tqdm(df_prompts.to_dict("records"), desc="Обработка изображений"):
        image_filename = row["image"]
        image_path = img_dir / image_filename

        # Проверка существования файла изображения
        if not image_path.exists():
            logger.warning(f"Изображение не найдено: {image_path}. Пропускаю.")
            continue

        try:
            #print(f"[DEBUG] type(_chunks): {type(_chunks)}")
            #print(f"[DEBUG] _chunks is None: {_chunks is None}")
            #print(f"[DEBUG] [1] Открываем изображение: {image_path}")
            # Открываем изображение
            img_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Ошибка при открытии изображения {image_path}: {e}")
            continue


        try:
            # Подготавливаем позитивные и негативные фрагменты для SigLIP
            # Если chunk_size=None, _make_chunks -> _chunks -> подставит MAX_SIG_TOK для SigLIP
            #print(f"[DEBUG] [2] Подготавливаем чанки SigLIP...")
            positive_chunks_siglip = _make_chunks(row["prompt"], row["prompt2"])
            #print(f"[DEBUG] [2.OK] positive_chunks_siglip: {positive_chunks_siglip}")
            negative_chunks_siglip = _make_chunks(row["negative"], row["negative2"])
            #print(f"[DEBUG] [2.OK] negative_chunks_siglip: {negative_chunks_siglip}")
            # --- Вычисление метрик ---

            #print(f"[DEBUG] [3] Начинаем SigLIP...")
            # --- SigLIP Score ---
            siglip_score = calculate_net_score(
                get_score_func=get_siglip_score,
                img=img_pil,
                positive_prompts=positive_chunks_siglip,
                negative_prompts=negative_chunks_siglip
            )
            #print(f"[DEBUG] [3.OK] siglip_score: {siglip_score}")

            #print(f"[DEBUG] [4] Начинаем Florence...")
            # --- Florence Score ---
            # Объединяем позитивные/негативные промпты в одну строку для Florence
            florence_positive_text = ", ".join(filter(None, [row["prompt"], row["prompt2"]]))
            florence_negative_text = ", ".join(filter(None, [row["negative"], row["negative2"]]))

            # Разбиваем объединённые тексты на фрагменты (используя chunk_size, которое может быть None)
            # Если chunk_size=None, Florence получит фрагмент [full_text] (так как _chunks(None) -> [full_text] после подстановки MAX_SIG_TOK и проверки длины)
            florence_pos_chunks = _make_chunks(florence_positive_text)
            #print(f"[DEBUG] [4.OK] florence_pos_chunks: {florence_pos_chunks}")
            florence_neg_chunks = _make_chunks(florence_negative_text)
            #print(f"[DEBUG] [4.OK] florence_neg_chunks: {florence_neg_chunks}")
            # Вычисляем средний Florence-скор для позитивных фрагментов
            florence_pos_scores = [
                get_florence_score(img_pil, chunk) for chunk in florence_pos_chunks
            ]
            #print(f"[DEBUG] [4.OK] florence_pos_scores: {florence_pos_scores}")
            avg_florence_pos_score = (
                sum(florence_pos_scores) / len(florence_pos_scores)
                if florence_pos_scores else 0.0
            )

            # Вычисляем средний Florence-скор для негативных фрагментов
            florence_neg_scores = [
                get_florence_score(img_pil, chunk) for chunk in florence_neg_chunks
            ]
            avg_florence_neg_score = (
                sum(florence_neg_scores) / len(florence_neg_scores)
                if florence_neg_scores else 0.0
            )
            #print(f"[DEBUG] [4.OK] florence_neg_scores: {florence_neg_scores}")
            florence_score = avg_florence_pos_score - avg_florence_neg_score
            #print(f"[DEBUG] [4.OK] florence_score: {florence_score}")

            #print(f"[DEBUG] [5] Начинаем IQA...")
            # --- IQA Score ---
            iqa_score = get_iqa(img_pil)
            #print(f"[DEBUG] [5.OK] iqa_score: {iqa_score}")

            #print(f"[DEBUG] [6] Начинаем DINO...")
            # --- DINO Score ---
            dino_score = get_dino(img_pil)
            #print(f"[DEBUG] [6.OK] dino_score: {dino_score}")

            # --- НОВАЯ МЕТРИКА: BLIP-2 Score ---
            # Используем те же позитивные чанки, что и для SigLIP
            blip2_score = get_blip2_match_score(img_pil, positive_chunks_siglip)
            # -------------------------------

            # --- Сохранение результатов для изображения ---
            results.append(
                {
                    "image": image_filename,
                    "sig": siglip_score,      # Исходное значение
                    "flor": florence_score,   # Исходное значение
                    "iqa": iqa_score,         # Исходное значение
                    "dino": dino_score,       # Исходное значение
                    "blip2": blip2_score,     # Исходное значение
                }
            )
            logger.debug(
                f"Изображение '{image_filename}': "
                f"SigLIP={siglip_score:.4f}, Florence={florence_score:.4f}, "
                f"IQA={iqa_score:.4f}, DINO={dino_score:.4f}"
            )
            #print(f"[DEBUG] [7] Успешно обработано: {image_filename}")

        except Exception as e:
            logger.error(
                f"Ошибка при вычислении метрик для изображения {image_path}: {e}"
            )
            import traceback
            traceback.print_exc() # Добавляем traceback для полной информации
            # Можно либо пропустить изображение, либо остановить процесс.
            # Здесь мы пропускаем и продолжаем.
            continue

    # --- 3. Проверка результатов ---
    if not results:
        error_msg = "Не удалось обработать ни одно изображение. Проверьте входные данные и логи."
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Успешно обработано {len(results)} изображений.")

    # --- 4. Нормализация метрик и вычисление итогового балла ---
    logger.info("Нормализую метрики и вычисляю итоговый балл...")
    
    # --- ЦЕНТРАЛИЗОВАННАЯ НОРМАЛИЗАЦИЯ ---
    # Определяем список метрик для нормализации
    metrics_to_normalize = ["sig", "flor", "iqa", "dino", "blip2"] # <-- Добавили "blip2"
    # При добавлении новой метрики, просто добавьте её имя в этот список:
    # metrics_to_normalize.append("new_metric")

    # Вызываем универсальную функцию нормализации
    normalized_data: Dict[str, np.ndarray] = normalize_metrics(results, metrics_to_normalize)

    # Обновляем словари в `results` нормализованными значениями
    # Используем ключи из normalized_data (например, "sig_norm")
    for metric_norm_name, norm_values in normalized_data.items():
        # Извлекаем оригинальное имя метрики (убираем "_norm")
        original_metric_name = metric_norm_name[:-5] # Убираем последние 5 символов "_norm"
        for i, res_dict in enumerate(results):
            res_dict[metric_norm_name] = norm_values[i]
            # Также можно обновить оригинальное значение, если это нужно
            # res_dict[original_metric_name] = norm_values[i] # Обычно этого не делают

    # --- Вычисление итогового балла ---
    for i, res_dict in enumerate(results):
        # Вычисляем итоговый балл как взвешенную сумму НОРМАЛИЗОВАННЫХ метрик
        # ВАЖНО: используем ключи с "_norm"
        total_score = (
            alpha * res_dict.get("sig_norm", 0.0)
            + beta * res_dict.get("flor_norm", 0.0)
            + gamma * res_dict.get("iqa_norm", 0.0)
            + delta * res_dict.get("dino_norm", 0.0)
            + epsilon * res_dict.get("blip2_norm", 0.0) # <-- Добавили BLIP-2
            # При добавлении новой метрики, добавьте её взвешенное нормированное значение:
            # + epsilon * res_dict.get("new_metric_norm", 0.0) 
        )
        # Нормализуем по сумме весов
        total_weight = alpha + beta + gamma + delta + epsilon # + epsilon # Обновите при добавлении
        if total_weight > 0:
            res_dict["total"] = total_score / total_weight
        else:
            res_dict["total"] = 0.0
    # --- Конец нормализации и вычисления итогового балла ---

    # --- 5. Сортировка и сохранение ---
    # Создаем DataFrame из результатов
    output_df = pd.DataFrame(results)
    # Сортируем по убыванию итогового балла
    output_df = output_df.sort_values(by="total", ascending=False).reset_index(drop=True)

    # Сохраняем результат в CSV файл в той же папке с изображениями
    output_file_path = img_dir / "ranking.csv"
    try:
        output_df.to_csv(output_file_path, index=False)
        logger.info(f"Результаты ранжирования сохранены в: {output_file_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла {output_file_path}: {e}")
        # Не останавливаем процесс из-за ошибки сохранения, просто логируем

    return output_df


# --- Шаблон для расширения ---
# Для добавления новой метрики:
# 1. Создайте функцию в `metrics.py` (см. `example_metric.py` как шаблон).
# 2. Импортируйте её в `ranking.py`.
# 3. Добавьте её вес в аргументы `rank_folder` и `config.py`.
# 4. Вызовите её в цикле обработки изображений.
# 5. Добавьте её имя в список `metrics_to_normalize` перед вызовом `utils.normalize_metrics`.
# 6. Добавьте её нормализованный вклад в расчет `total_score`.
# 7. Добавьте её вес в аргументы CLI (`cli.py`).
# --- Конец шаблона ---