# src/rank_images/cli.py
"""
Интерфейс командной строки (CLI) для проекта rank_images.

Этот модуль предоставляет удобный способ запуска процесса ранжирования
изображений из терминала. Он обрабатывает аргументы командной строки,
загружает модели и вызывает основную функцию ранжирования.
"""

import logging
import sys



# --- Настройка логирования ДО любых других импортов ---
# Отключаем базовую конфигурацию, чтобы установить свою
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.NOTSET)

# Создаем и настраиваем корневой логгер
root_logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)
root_logger.setLevel(logging.INFO) # Уровень по умолчанию для всего

# --- Отключаем/ограничиваем логи шумных библиотек ---
noisy_loggers = [
    "urllib3", "PIL", "matplotlib", "transformers", "torch", "torchvision",
    "torchaudio", "tokenizers", "datasets", "huggingface_hub", "filelock",
    "fsspec", "asyncio", "openai", "httpx", "httpcore", "tensorflow", "timm"
]
for lib_name in noisy_loggers:
    logging.getLogger(lib_name).setLevel(logging.WARNING)
# --- Конец настройки логирования ---


import argparse
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# Импорт основной логики ранжирования
from .ranking import rank_folder
# Импорт загрузчика моделей
from .models import load_models, METRIC_TO_MODELS # <-- Импортируем карту
# Импорт конфигурации по умолчанию
from .config import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    EPSILON_DEFAULT,
    ZETA_DEFAULT,
    THETA_DEFAULT, # <-- НОВОЕ
)
# --- ИМПОРТ МОДУЛЯ КОНФИГУРАЦИИ ---
from .pipeline_config import load_pipeline_config, get_default_weights, get_chunk_size, get_enabled_metrics
# --- Настройка логгирования для CLI ---
# Базовая настройка логгера для вывода в консоль
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] # Вывод в stdout
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Основная точка входа для CLI.

    Обрабатывает аргументы командной строки, загружает модели и
    запускает процесс ранжирования.
    """
    # --- АРГУМЕНТЫ ДЛЯ ВЕСОВ ---
    # Эти аргументы позволяют переопределить веса из JSON-конфига
    parser = argparse.ArgumentParser(
        description=(
            "Ранжирует изображения в заданной директории на основе "
            "текстовых промптов и внутреннего качества, используя "
            f"{', '.join(METRIC_TO_MODELS.keys())}." # <-- Используем ключи из карты
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "img_dir", 
        nargs='?', 
        type=Path, 
        default=None,
        help="Путь к директории с изображениями для ранжирования. Обязателен, если не указан --demo."
        ),
        
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help=(
            "Источник текстовых промптов. Может быть:\n"
            "- Путь к .json файлу с ключами 'prompt', 'prompt2', 'negative', 'negative2'.\n"
            "- Произвольная текстовая строка (будет использована как 'prompt').\n"
            "Если не указан, ранжирование будет основано только на именах файлов."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=f"Вес метрики SigLIP (схожесть изображения и текста). По умолчанию {ALPHA_DEFAULT}.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help=f"Вес метрики Florence-2 (поиск объектов по запросу). По умолчанию {BETA_DEFAULT}.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=f"Вес метрики CLIP-IQA (общее качество изображения). По умолчанию {GAMMA_DEFAULT}.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help=f"Вес метрики DINOv2 (внутренние признаки изображения). По умолчанию {DELTA_DEFAULT}.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=f"Вес метрики BLIP-2 (Image-Text Matching). По умолчанию {EPSILON_DEFAULT}.",
    )

    parser.add_argument(
        "--zeta", 
        type=float,
        default=None, 
        help=f"Вес метрики BLIP Caption + BERTScore к prompt. По умолчанию берётся из JSON-конфига или {ZETA_DEFAULT}.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=THETA_DEFAULT,
        help=f"Вес метрики BLIP-2 Caption + BERTScore к prompt. По умолчанию {THETA_DEFAULT}.",
    )
    parser.add_argument("--phi", type=float, default=0.4,
                    help="Вес ImageReward (imr)")

    #--- НОВЫЙ АРГУМЕНТ ---
    parser.add_argument(
        "--pipeline-config",
        type=str, # <-- Принимаем путь к файлу как строку
        default=None,
        help=(
            "Путь к JSON-файлу конфигурации пайплайна. "
            "Определяет, какие метрики включены и их веса по умолчанию. "
            "Если не указан, используется стандартная конфигурация."
        ),
    )
    # ----------------------
    parser.add_argument(
        "--chunk",
        type=int,
        default=None,
        help=(
            "Максимальное количество токенов в одном фрагменте текста для SigLIP.\n"
            "Используется для разбиения длинных текстов. По умолчанию используется значение из config.py."
        ),
    )
    # Добавление аргумента для удобного запуска демонстрации
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Запустить демонстрацию на встроенных примерах из data/demo_images/.",
    )

    # --- Парсинг аргументов ---
    args = parser.parse_args()

    # --- Загрузка и валидация JSON-конфигурации пайплайна ---
    pipeline_config = load_pipeline_config(args.pipeline_config)
    # Извлекаем веса по умолчанию из конфига
    default_weights_from_config = get_default_weights(pipeline_config)
    # Извлекаем chunk_size из конфига
    chunk_size_from_config = get_chunk_size(pipeline_config)
    # ------------------------------

    # --- Определение финальных значений весов ---
    # Если вес НЕ задан через CLI (--alpha None), используем значение из JSON-конфига или дефолтное
    final_alpha = args.alpha if args.alpha is not None else default_weights_from_config.get("alpha", ALPHA_DEFAULT)
    final_beta = args.beta if args.beta is not None else default_weights_from_config.get("beta", BETA_DEFAULT)
    final_gamma = args.gamma if args.gamma is not None else default_weights_from_config.get("gamma", GAMMA_DEFAULT)
    final_delta = args.delta if args.delta is not None else default_weights_from_config.get("delta", DELTA_DEFAULT)
    final_epsilon = args.epsilon if args.epsilon is not None else default_weights_from_config.get("epsilon", EPSILON_DEFAULT)
    final_zeta = args.zeta if args.zeta is not None else default_weights_from_config.get("zeta", ZETA_DEFAULT) # <-- НОВОЕ
    final_theta = args.zeta if args.theta is not None else default_weights_from_config.get("theta", THETA_DEFAULT) # <-- НОВОЕ
    # --- Определение финального chunk_size ---
    # Приоритет: CLI > JSON-config > config.py
    final_chunk_size = args.chunk if args.chunk is not None else chunk_size_from_config
    # ------------------------------------------

    # --- Логика обработки аргументов ---
    # Если указан --demo, переопределяем img_dir и prompts
    if args.demo:
        # Определяем путь к корню проекта (предполагаем, что cli.py находится в src/rank_images/)
        # Это может быть не очень надежно, но работает для стандартной структуры
        project_root = Path(__file__).resolve().parent.parent.parent
        demo_images_dir = project_root / "data" / "demo_images"
        
        if not demo_images_dir.exists():
            logger.error(f"Демонстрационная папка не найдена: {demo_images_dir}")
            sys.exit(1)
            
        args.img_dir = demo_images_dir
        args.prompts = str(demo_images_dir / "prompts.json") # <-- Преобразование в строку
        logger.info("Запуск в демонстрационном режиме.")
        logger.info(f"Папка с изображениями: {args.img_dir}")
        logger.info(f"Файл с промптами: {args.prompts}")

    # --- Валидация путей ---
    if not args.img_dir.exists() or not args.img_dir.is_dir():
        logger.error(f"Указанная директория изображений не существует или не является папкой: {args.img_dir}")
        sys.exit(1)

    # prompts может быть None, путем к файлу или строкой. Валидация будет внутри build_dataframe.
    if args.prompts and Path(args.prompts).suffix.lower() == ".json":
        prompts_path = Path(args.prompts)
        if not prompts_path.exists():
             logger.error(f"Указанный файл промптов не найден: {prompts_path}")
             sys.exit(1)

    # --- Загрузка моделей ---
    logger.info("Загружаю модели (на CPU)...")
    try:
        # Извлекаем список включённых метрик из конфига
        pipeline_config = load_pipeline_config(args.pipeline_config)
        enabled_metrics_list = get_enabled_metrics(pipeline_config)
        
        # Передаём список в load_models для оптимизированной загрузки
        load_models(enabled_metrics_list) # <-- Передаём список
        logger.info("Все модели успешно загружены.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей: {e}")
        sys.exit(1)

    # --- Запуск ранжирования ---
    logger.info("Начинаю процесс ранжирования...")
    try:
        result_df: pd.DataFrame = rank_folder(
            img_dir=args.img_dir,
            prompts_in=args.prompts,
            # --- ПЕРЕДАЁМ ФИНАЛЬНЫЕ ЗНАЧЕНИЯ ---
            alpha=final_alpha,
            beta=final_beta,
            gamma=final_gamma,
            delta=final_delta,
            epsilon=final_epsilon,
            zeta=final_zeta, 
            theta=final_theta, 
            # ----------------------------------
            chunk_size=final_chunk_size, # <-- Обновлённый chunk_size
            # --- ПЕРЕДАЁМ КОНФИГУРАЦИЮ ПАЙПЛАЙНА ---
            pipeline_config=pipeline_config, 
            # --------------------------------------
        )

        logger.info("Процесс ранжирования завершён успешно.")
        # Выводим первые несколько строк результата в консоль
        print("\n--- Результаты ранжирования (первые 5 строк) ---")
        print(result_df.head())
        print("-----------------------------------------------")
        print(f"Полный результат сохранён в: {args.img_dir / 'ranking.csv'}")

    except Exception as e:
        logger.error(f"Ошибка во время ранжирования: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()