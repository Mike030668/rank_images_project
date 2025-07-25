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
from .utils import normalize_metrics, calculate_net_score
from .data_processing import build_dataframe, _chunks
from .metrics import (
    get_siglip_score,
    get_florence_score,
    get_iqa,
    get_dino,
    get_blip2_match_score,
    #get_blip_caption_bertscore
)
# Импорт конфигурации
from .config import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    # --- НОВОЕ ---
    EPSILON_DEFAULT,
    ZETA_DEFAULT,
    # ------------
    MAX_SIG_TOK  # Максимальное количество токенов для SigLIP
)

# Настройка логгирования
logger = logging.getLogger(__name__)

# --- ИМПОРТ НОВОГО МОДУЛЯ КОНФИГУРАЦИИ ---
from .pipeline_config import get_enabled_metrics # <-- НОВОЕ
# --
def rank_folder(
    img_dir: Path,
    prompts_in: Optional[Union[str, dict, pd.DataFrame]] = None,
    alpha: float = ALPHA_DEFAULT,
    beta: float = BETA_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
    delta: float = DELTA_DEFAULT,
    epsilon: float = EPSILON_DEFAULT, 
    zeta: float = ZETA_DEFAULT, #
    chunk_size: Optional[int] = None,  # Может быть None
    # --- НОВЫЙ АРГУМЕНТ ---
    pipeline_config: Optional[Dict[str, Any]] = None, # <-- НОВОЕ
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
        epsilon (float): Вес метрики BLIP-2 (Image-Text Matching).
                         По умолчанию 0.3.
        zeta (float): Вес метрики BLIP Caption + BERTScore к prompt.
                      По умолчанию 0.0.               
        chunk_size (int | None): Максимальное количество токенов в одном
                                 фрагменте текста для SigLIP. Если None,
                                 используется значение по умолчанию из config.
        pipeline_config (dict | None): Конфигурация пайплайна, определяющая
                                       включённые метрики и их веса по умолчанию.
                                       Если None, используются аргументы CLI.

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
            # Инициализируем все возможные метрики нулями
            sig_score = 0.0
            flor_score = 0.0
            iqa_score = 0.0
            dino_score = 0.0
            blip2_score = 0.0
            blip_cap_score = 0.0 # 

            #print(f"[DEBUG] [3] Начинаем SigLIP...")
            # --- SigLIP Score ---
            if "sig" in enabled_metrics_list:
                siglip_pos_score = get_siglip_score(img_pil, positive_chunks_siglip)
                siglip_neg_score = get_siglip_score(img_pil, negative_chunks_siglip)
                sig_score = siglip_pos_score - siglip_neg_score
                logger.debug(f"  SigLIP Score: {sig_score:.4f}")

            # --- Florence Score ---
            if "flor" in enabled_metrics_list:
                florence_positive_text = ", ".join(filter(None, [row["prompt"], row["prompt2"]]))
                florence_negative_text = ", ".join(filter(None, [row["negative"], row["negative2"]]))
                
                florence_pos_chunks = _make_chunks(florence_positive_text)
                florence_neg_chunks = _make_chunks(florence_negative_text)

                florence_pos_scores = [
                    get_florence_score(img_pil, chunk) for chunk in florence_pos_chunks
                ]
                avg_florence_pos_score = (
                    sum(florence_pos_scores) / len(florence_pos_scores)
                    if florence_pos_scores else 0.0
                )

                florence_neg_scores = [
                    get_florence_score(img_pil, chunk) for chunk in florence_neg_chunks
                ]
                avg_florence_neg_score = (
                    sum(florence_neg_scores) / len(florence_neg_scores)
                    if florence_neg_scores else 0.0
                )

                flor_score = avg_florence_pos_score - avg_florence_neg_score
                logger.debug(f"  Florence Score: {flor_score:.4f}")

            # --- IQA Score ---
            if "iqa" in enabled_metrics_list:
                iqa_score = get_iqa(img_pil)
                logger.debug(f"  IQA Score: {iqa_score:.4f}")

            # --- DINO Score ---
            if "dino" in enabled_metrics_list:
                dino_score = get_dino(img_pil)
                logger.debug(f"  DINO Score: {dino_score:.4f}")

            # --- BLIP-2 Score ---
            if "blip2" in enabled_metrics_list:
                # Используем те же позитивные чанки, что и для SigLIP
                blip2_pos_score = get_blip2_match_score(img_pil, positive_chunks_siglip)
                blip2_neg_score = get_blip2_match_score(img_pil, negative_chunks_siglip)
                blip2_score = blip2_pos_score - blip2_neg_score # <-- Штрафуем на негативные фрагменты
            # -------------------------------
                logger.debug(f"  BLIP-2 Score: {blip2_score:.4f}")

            # --- BLIP Caption + BERTScore ---
            #if "blip_cap" in enabled_metrics_list:
                # Используем основной промпт
                #blip_cap_score = get_blip_caption_bertscore(img_pil, row["prompt"])
                #logger.debug(f"  BLIP Caption Score: {blip_cap_score:.4f}")
            # -------------------------------

            # --- Сохранение результатов для изображения ---
            results.append(
                {
                    "image": image_filename,
                    "sig": sig_score,      # Исходное значение
                    "flor": flor_score,   # Исходное значение
                    "iqa": iqa_score,         # Исходное значение
                    "dino": dino_score,       # Исходное значение
                    "blip2": blip2_score,     # Исходное значение
                   # "blip_cap": blip_cap_score, # <-- НОВОЕ
                }
            )
            logger.debug(
                f"Изображение '{image_filename}': "
                f"SigLIP={sig_score:.4f}, Florence={flor_score:.4f}, "
                f"IQA={iqa_score:.4f}, DINO={dino_score:.4f}, BLIP2={blip2_score:.4f}, "
                f"BLIP_Caption={blip_cap_score:.4f}" # <-- Обновлено
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
    # Определяем список метрик для нормализации на основе включённых
    metrics_to_normalize = enabled_metrics_list

    # Вызываем универсальную функцию нормализации
    normalized_data: Dict[str, np.ndarray] = normalize_metrics(results, metrics_to_normalize)

    # Обновляем словари в `results` нормализованными значениями
    for metric_norm_name, norm_values in normalized_data.items():
        # metric_norm_name будет, например, "sig_norm"
        for i, res_dict in enumerate(results):
            res_dict[metric_norm_name] = norm_values[i]

    # --- Вычисление итогового балла ---
    # Сначала соберём все возможные веса
    weight_map = {
        "sig": alpha,
        "flor": beta,
        "iqa": gamma,
        "dino": delta,
        "blip2": epsilon,
        "blip_cap": zeta, # <-- НОВОЕ
    }

    # --- Вычисление итогового балла ---
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
# Для добавления новой метрики:
# 1. Создайте функцию в `metrics.py` (см. `example_metric.py` как шаблон).
# 2. Импортируйте её в `ranking.py`.
# 3. Добавьте её вес в аргументы `rank_folder` и `config.py`.
# 4. Вызовите её в цикле обработки изображений.
# 5. Добавьте её имя в список `metrics_to_normalize` перед вызовом `utils.normalize_metrics`.
# 6. Добавьте её нормализованный вклад в расчет `total_score`.
# 7. Добавьте её вес в аргументы CLI (`cli.py`).
# --- Конец шаблона ---