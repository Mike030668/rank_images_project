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
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# Импорт вспомогательных функций и компонентов
from .data_processing import build_dataframe, _chunks, _z
# Импорт метрик
from .metrics import (
    get_siglip_score,
    get_florence_score,
    get_iqa,
    get_dino,
    get_blip2_match_score,
    # --- НОВАЯ МЕТРИКА ---
    get_blip_caption_bertscore,
    # --------------------
)
# Импорт утилит
# --- ИМПОРТ ДЛЯ ПАЙПЛАЙНА ---
from .utils import normalize_metrics
from .pipeline_config import get_enabled_metrics
# ----------------------------
# Импорт конфигурации
from .config import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    EPSILON_DEFAULT,
    # --- НОВАЯ МЕТРИКА ---
    ZETA_DEFAULT,
    # --------------------
)

logger = logging.getLogger(__name__)

def rank_folder(
    img_dir: Path,
    prompts_in: Optional[Union[str, dict, pd.DataFrame]] = None,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    delta: float = DELTA_DEFAULT,
    epsilon: float = EPSILON_DEFAULT,
    # --- НОВАЯ МЕТРИКА ---
    zeta: float = ZETA_DEFAULT,
    # --------------------
    chunk_size: Optional[int] = None,
    # --- ПАЙПЛАЙН ---
    pipeline_config: Optional[Dict[str, Any]] = None,
    # ----------------
) -> pd.DataFrame:
    """
    Ранжирует изображения в заданной директории на основе текстовых промптов
    и внутреннего качества, используя SigLIP-2, Florence-2, CLIP-IQA, DINOv2,
    BLIP-2 и BLIP Caption + BERTScore.

    Алгоритм:
    1. Строит DataFrame с изображениями и промптами.
    2. Для каждого изображения:
       a. Вычисляет метрики SigLIP, Florence, IQA, DINO, BLIP-2, BLIP Caption.
       b. Нормализует значения метрик по Z-оценке.
       c. Вычисляет итоговый балл как взвешенную сумму нормализованных метрик.
    3. Сортирует изображения по убыванию итогового балла.
    4. Сохраняет результат в `img_dir/ranking.csv`.

    Args:
        img_dir (Path): Путь к директории с изображениями для ранжирования.
        prompts_in (str | dict | pd.DataFrame | None): Источник текстовых промптов.
            - None: Используются только имена файлов.
            - str (путь к .json): JSON-файл с ключами 'prompt', 'prompt2', 'negative', 'negative2'.
            - dict: Словарь с ключами 'prompt', 'prompt2', 'negative', 'negative2'.
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
        epsilon (float): Вес метрики BLIP-2 (Image-Text Matching).
                         По умолчанию 0.3.
        # --- НОВАЯ МЕТРИКА ---
        zeta (float): Вес метрики BLIP Caption + BERTScore к prompt.
                      По умолчанию 0.25.
        # --------------------
        chunk_size (int | None): Максимальное количество токенов в одном фрагменте
                                 текста для SigLIP.
                                 Используется для разбиения длинных текстов. По
                                 умолчанию используется значение из config.py.
        # --- ПАЙПЛАЙН ---
        pipeline_config (dict | None): Конфигурация пайплайна, определяющая
                                       включённые метрики и их веса по умолчанию.
                                       Если None, используются аргументы CLI.
        # ----------------

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
    
    # --- ИЗВЛЕЧЕНИЕ КОНФИГУРАЦИИ ПАЙПЛАЙНА ---
    enabled_metrics_list = get_enabled_metrics(pipeline_config) if pipeline_config else []
    logger.info(f"Включённые метрики: {enabled_metrics_list}")
    # ------------------------------------------

    # --- 1. Подготовка данных ---
    df_prompts = build_dataframe(img_dir, prompts_in)
    results = [] # Список для хранения результатов по каждому изображению
    logger.info("Начинаю вычисление метрик для изображений...")

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
            # Открываем изображение
            img_pil = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Ошибка при открытии изображения {image_path}: {e}")
            continue

        # --- Подготовка текстовых запросов ---
        # Функция для создания списка непустых фрагментов текста
        def _make_chunks(*texts: str) -> List[str]:
            output_chunks = []
            for text in texts:
                # Разбиваем каждый текст на фрагменты и добавляем непустые
                output_chunks.extend(_chunks(text, chunk_size))
            # Возвращаем список уникальных непустых фрагментов
            return [chunk for chunk in output_chunks if chunk.strip()]

        # Подготавливаем позитивные и негативные фрагменты для SigLIP
        positive_chunks_siglip = _make_chunks(row["prompt"], row["prompt2"])
        negative_chunks_siglip = _make_chunks(row["negative"], row["negative2"])

        # --- Вычисление метрик ---
        try:
            # --- SigLIP Score ---
            siglip_pos_score = get_siglip_score(img_pil, positive_chunks_siglip) if "sig" in enabled_metrics_list else 0.0
            siglip_neg_score = get_siglip_score(img_pil, negative_chunks_siglip) if "sig" in enabled_metrics_list else 0.0
            siglip_score = siglip_pos_score - siglip_neg_score if "sig" in enabled_metrics_list else 0.0

            # --- Florence Score ---
            # Объединяем позитивные/негативные промпты в одну строку для Florence
            florence_positive_text = ", ".join(filter(None, [row["prompt"], row["prompt2"]]))
            florence_negative_text = ", ".join(filter(None, [row["negative"], row["negative2"]]))
            
            # Разбиваем объединённые тексты на фрагменты (обычно 1 фрагмент)
            florence_pos_chunks = _make_chunks(florence_positive_text)
            florence_neg_chunks = _make_chunks(florence_negative_text)

            # Вычисляем средний Florence-скор для позитивных фрагментов
            florence_pos_scores = [
                get_florence_score(img_pil, chunk) for chunk in florence_pos_chunks
            ] if "flor" in enabled_metrics_list else []
            avg_florence_pos_score = (
                sum(florence_pos_scores) / len(florence_pos_scores)
                if florence_pos_scores else 0.0
            )

            # Вычисляем средний Florence-скор для негативных фрагментов
            florence_neg_scores = [
                get_florence_score(img_pil, chunk) for chunk in florence_neg_chunks
            ] if "flor" in enabled_metrics_list else []
            avg_florence_neg_score = (
                sum(florence_neg_scores) / len(florence_neg_scores)
                if florence_neg_scores else 0.0
            )

            florence_score = avg_florence_pos_score - avg_florence_neg_score if "flor" in enabled_metrics_list else 0.0

            # --- IQA Score ---
            iqa_score = get_iqa(img_pil) if "iqa" in enabled_metrics_list else 0.0

            # --- DINO Score ---
            dino_score = get_dino(img_pil) if "dino" in enabled_metrics_list else 0.0

            # --- BLIP-2 Score ---
            # Используем те же позитивные чанки, что и для SigLIP
            blip2_pos_score = get_blip2_match_score(img_pil, positive_chunks_siglip) if "blip2" in enabled_metrics_list else 0.0
            blip2_neg_score = get_blip2_match_score(img_pil, negative_chunks_siglip) if "blip2" in enabled_metrics_list else 0.0
            blip2_score = blip2_pos_score - blip2_neg_score if "blip2" in enabled_metrics_list else 0.0

            # --- НОВАЯ МЕТРИКА: BLIP Caption + BERTScore ---
            # Используем основной промпт
            blip_caption_score = get_blip_caption_bertscore(img_pil, row["prompt"]) if "blip_cap" in enabled_metrics_list else 0.0
            # ------------------------------------------------

            # --- Сохранение результатов для изображения ---
            # Сохраняем ИСХОДНЫЕ (не нормализованные) значения метрик
            results.append(
                {
                    "image": image_filename,
                    "sig": siglip_score,      # Исходное значение
                    "flor": florence_score,   # Исходное значение
                    "iqa": iqa_score,         # Исходное значение
                    "dino": dino_score,       # Исходное значение
                    "blip2": blip2_score,     # Исходное значение
                    # --- НОВАЯ МЕТРИКА ---
                    "blip_cap": blip_caption_score, # Исходное значение
                    # --------------------
                }
            )
            logger.debug(
                f"Изображение '{image_filename}': "
                f"SigLIP={siglip_score:.4f}, Florence={florence_score:.4f}, "
                f"IQA={iqa_score:.4f}, DINO={dino_score:.4f}, BLIP2={blip2_score:.4f}, "
                f"BLIP_Caption={blip_caption_score:.4f}" # <-- Обновлено
            )

        except Exception as e:
            logger.error(
                f"Ошибка при вычислении метрик для изображения {image_path}: {e}"
            )
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
    # Определяем список метрик для нормализации на основе включённых
    metrics_to_normalize = [m for m in ["sig", "flor", "iqa", "dino", "blip2", "blip_cap"] if m in enabled_metrics_list]

    # Вызываем универсальную функцию нормализации
    normalized_data: Dict[str, np.ndarray] = normalize_metrics(results, metrics_to_normalize)

    # Обновляем словари в `results` нормализованными значениями
    for metric_norm_name, norm_values in normalized_data.items():
        # metric_norm_name будет, например, "sig_norm"
        for i, res_dict in enumerate(results):
            res_dict[metric_norm_name] = norm_values[i]

    # --- Вычисление итогового балла ---
    # Собираем веса для включённых метрик
    weight_map = {
        "sig": alpha,
        "flor": beta,
        "iqa": gamma,
        "dino": delta,
        "blip2": epsilon,
        "blip_cap": zeta, # <-- НОВАЯ МЕТРИКА
    }
    
    for i, res_dict in enumerate(results):
        # Вычисляем итоговый балл как взвешенную сумму НОРМАЛИЗОВАННЫХ метрик
        # Учитываем только включённые метрики
        total_score = 0.0
        total_weight = 0.0
        for metric_abbr, weight in weight_map.items():
            if metric_abbr in enabled_metrics_list:
                norm_key = f"{metric_abbr}_norm"
                total_score += weight * res_dict.get(norm_key, 0.0)
                total_weight += weight
        
        # Нормализуем по сумме используемых весов
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
# При добавлении новой метрики `get_new_metric_score`:
# 1. Создайте функцию в `metrics.py` (см. `example_metric.py` как шаблон).
# 2. Импортируйте её в `ranking.py`.
# 3. Добавьте её вес в аргументы `rank_folder` и `config.py`.
# 4. Вызовите её в цикле обработки изображений.
# 5. Добавьте её имя в список `metrics_to_normalize` перед вызовом `utils.normalize_metrics`.
# 6. Добавьте её нормализованный вклад в вычисление `total_score`,
#    используя новый вес (например, `zeta`).
# 7. Добавьте её вес в аргументы CLI (`cli.py`).
# --- Конец шаблона ---