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
    get_blip2_caption_bertscore, # <-- НОВОЕ
    # --------------------
)
# --- ИМПОРТ УТИЛИТ ДЛЯ ПАЙПЛАЙНА ---
from .utils import normalize_metrics
from .pipeline_config import get_enabled_metrics, get_all_metrics
# -----------------------------------
# Импорт конфигурации
from .config import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    EPSILON_DEFAULT,
    # --- НОВАЯ МЕТРИКА ---
    ZETA_DEFAULT,
    THETA_DEFAULT, # <-- НОВОЕ
    # --------------------
    ALL_METRICS
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
    theta: float = THETA_DEFAULT, # <-- НОВОЕ
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
    # Получаем список ВСЕХ доступных метрик из конфига
    all_metrics_list = get_all_metrics(pipeline_config) if pipeline_config else []
    logger.debug(f"Все доступные метрики: {all_metrics_list}")
    # ------------------------------------------

    # --- 1. Подготовка данных ---
    df_prompts = build_dataframe(img_dir, prompts_in)
    results = [] # Список для хранения результатов по каждому изображению
    logger.info("Начинаю вычисление метрик для изображений...")

    # --- 2. Цикл по изображениям ---
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

        
        try:
            # Подготавливаем позитивные и негативные фрагменты для SigLIP
            positive_chunks_siglip = _make_chunks(row["prompt"], row["prompt2"])
            negative_chunks_siglip = _make_chunks(row["negative"], row["negative2"])

            # --- Вычисление метрик ---
            # --- Создаём пустой словарь для результатов текущего изображения ---
            res_dict = {"image": image_filename}
            
            # --- Итерируемся по ВКЛЮЧЕННЫМ метрикам ---
            for metric_name in enabled_metrics_list:
                try:
                    if metric_name == "sig":
                        siglip_pos_score = get_siglip_score(img_pil, positive_chunks_siglip)
                        siglip_neg_score = get_siglip_score(img_pil, negative_chunks_siglip)
                        siglip_score = siglip_pos_score - siglip_neg_score
                        res_dict["sig"] = siglip_score
                        logger.debug(f"  SigLIP Score: {siglip_score:.4f}")

                    elif metric_name == "flor":
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

                        florence_score = avg_florence_pos_score - avg_florence_neg_score
                        res_dict["flor"] = florence_score
                        logger.debug(f"  Florence Score: {florence_score:.4f}")

                    elif metric_name == "iqa":
                        iqa_score = get_iqa(img_pil)
                        res_dict["iqa"] = iqa_score
                        logger.debug(f"  IQA Score: {iqa_score:.4f}")
                    

                    elif metric_name == "dino":
                        dino_score = get_dino(img_pil)
                        res_dict["dino"] = dino_score
                        logger.debug(f"  DINO Score: {dino_score:.4f}")

                    elif metric_name == "blip2":
                        blip2_pos_score = get_blip2_match_score(img_pil, positive_chunks_siglip)
                        blip2_neg_score = get_blip2_match_score(img_pil, negative_chunks_siglip)
                        blip2_score = blip2_pos_score - blip2_neg_score
                        res_dict["blip2"] = blip2_score
                        logger.debug(f"  BLIP-2 Score: {blip2_score:.4f}")

                    elif metric_name == "blip_cap":
                        blip_caption_score = get_blip_caption_bertscore(img_pil, row["prompt"])
                        res_dict["blip_cap"] = blip_caption_score
                        #logger.debug(f"  BLIP Caption Score: {blip_caption_score:.4f}")
                        logger.debug(f"[RANKING_DEBUG] blip_caption_score для {image_filename}: {blip_caption_score:.4f}")
                    
                    elif metric_name == "blip2_cap":
                         blip2_caption_score = get_blip2_caption_bertscore(img_pil, row["prompt"])
                         res_dict["blip2_cap"] = blip2_caption_score
                         #logger.debug(f"  BLIP-2 Caption Score: {blip2_caption_score:.4f}")
                         logger.debug(f"[RANKING_DEBUG] blip2_caption_score для {image_filename}: {blip2_caption_score:.4f}")

                    # --- Добавьте elif для новых метрик здесь ---
                    # elif metric_name == "new_metric":
                    #     new_metric_score = get_new_metric_score(...)
                    #     res_dict["new_metric_abbr"] = new_metric_score
                    #     logger.debug(f"  New Metric Score: {new_metric_score:.4f}")
                    # ---------------------------------------------

                    else:
                        logger.warning(f"Неизвестная метрика '{metric_name}' в списке включённых. Пропускаю.")

                except Exception as metric_e:
                    logger.error(f"Ошибка при вычислении метрики '{metric_name}' для изображения {image_path}: {metric_e}")
                    # Можно либо пропустить метрику, либо записать 0.0
                    res_dict[metric_name] = 0.0 # Записываем 0.0 в случае ошибки
            
            # --- Конец итерации по включённым метрикам ---

            # --- Сохранение результатов для изображения ---
            results.append(res_dict)
            logger.debug(
                f"Изображение '{image_filename}': "
                f"{', '.join([f'{k}={v:.4f}' for k, v in res_dict.items() if k != 'image'])}"
            )

        except Exception as e:
            logger.error(
                f"Ошибка при вычислении метрик для изображения {image_path}: {e}"
            )
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
    # Определяем список метрик для нормализации на основе ВКЛЮЧЕННЫХ
    # Это пересечение всех доступных и включённых метрик
    metrics_to_normalize = [m for m in all_metrics_list if m in enabled_metrics_list]
    logger.debug(f"Метрики для нормализации: {metrics_to_normalize}")
    
    # Вызываем универсальную функцию нормализации
    normalized_data: Dict[str, np.ndarray] = normalize_metrics(results, metrics_to_normalize)

    # Обновляем словари в `results` нормализованными значениями
    for metric_norm_name, norm_values in normalized_data.items():
        # metric_norm_name будет, например, "sig_norm"
        for i, res_dict in enumerate(results):
            res_dict[metric_norm_name] = norm_values[i]
    # --- Конец централизованной нормализации ---

    # --- Вычисление итогового балла ---
    # --- ИСПОЛЬЗУЕМ УЖЕ ПЕРЕДАННЫЕ ФИНАЛЬНЫЕ ВЕСА ---
    # alpha, beta, gamma, delta, epsilon, zeta, theta уже содержат
    # значения, определённые по приоритету: CLI > JSON-Config > config.py defaults
    # ----------------------------------------------

    for i, res_dict in enumerate(results):
        # Вычисляем итоговый балл как взвешенную сумму НОРМАЛИЗОВАННЫХ метрик
        # Учитываем только ВКЛЮЧЕННЫЕ метрики
        total_score = 0.0
        total_weight = 0.0
        
        # --- Используем переданные аргументы напрямую ---
        if "sig" in enabled_metrics_list:
            total_score += alpha * res_dict.get("sig_norm", 0.0)
            total_weight += alpha
        if "flor" in enabled_metrics_list:
            total_score += beta * res_dict.get("flor_norm", 0.0)
            total_weight += beta
        if "iqa" in enabled_metrics_list:
            total_score += gamma * res_dict.get("iqa_norm", 0.0)
            total_weight += gamma
        if "dino" in enabled_metrics_list:
            total_score += delta * res_dict.get("dino_norm", 0.0)
            total_weight += delta
        if "blip2" in enabled_metrics_list:
            total_score += epsilon * res_dict.get("blip2_norm", 0.0)
            total_weight += epsilon
        if "blip_cap" in enabled_metrics_list:
            total_score += zeta * res_dict.get("blip_cap_norm", 0.0)
            total_weight += zeta
        # --- НОВАЯ МЕТРИКА ---
        if "blip2_cap" in enabled_metrics_list:
            total_score += theta * res_dict.get("blip2_cap_norm", 0.0)
            total_weight += theta
        # --------------------
        # --- Шаблон для добавления новой метрики ---
        # if "new_metric" in enabled_metrics_list:
        #     total_score += new_metric_weight_arg * res_dict.get("new_metric_norm", 0.0)
        #     total_weight += new_metric_weight_arg
        # --------------------------------------------
        
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
# 1. Добавьте её импорт вверху файла.
# 2. Добавьте новый аргумент веса (например, `zeta: float = ZETA_DEFAULT`)
#    в сигнатуру функции `rank_folder`.
# 3. В цикле обработки изображений:
#    a. Вызовите `new_metric_score = get_new_metric_score(img_pil, ...)`.
#    b. Добавьте `"new_metric": new_metric_score` в словарь `results`.
# 4. После цикла:
#    a. Преобразуйте значения в массив NumPy.
#    b. Нормализуйте с помощью `_z(...)`.
#    c. Обновите словари в `results` нормализованным значением.
#    d. Добавьте нормализованное значение в вычисление `total_score`,
#       используя новый вес `zeta`.
# 5. Не забудьте обновить документацию функции `rank_folder`.
# --- Конец шаблона ---