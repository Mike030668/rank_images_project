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
from .models import load_models
# Импорт конфигурации по умолчанию
from .config import (
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    EPSILON_DEFAULT,
)

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
    # --- Определение и парсинг аргументов ---
    parser = argparse.ArgumentParser(
        description=(
            "Ранжирует изображения в заданной директории на основе "
            "текстовых промптов и внутреннего качества, используя "
            "SigLIP-2, Florence-2, CLIP-IQA и DINOv2."
        ),
        formatter_class=argparse.RawTextHelpFormatter, # Для корректного отображения \n в help
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
        default=ALPHA_DEFAULT,
        help=f"Вес метрики SigLIP (схожесть изображения и текста). По умолчанию {ALPHA_DEFAULT}.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=BETA_DEFAULT,
        help=f"Вес метрики Florence-2 (поиск объектов по запросу). По умолчанию {BETA_DEFAULT}.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA_DEFAULT,
        help=f"Вес метрики CLIP-IQA (общее качество изображения). По умолчанию {GAMMA_DEFAULT}.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DELTA_DEFAULT,
        help=f"Вес метрики DINOv2 (внутренние признаки изображения). По умолчанию {DELTA_DEFAULT}.",
    )
    # --- НОВЫЙ АРГУМЕНТ ---
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON_DEFAULT,
        help=f"Вес метрики BLIP-2 (Image-Text Matching). По умолчанию {EPSILON_DEFAULT}.",
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
        load_models()
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
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            epsilon=args.epsilon, # <-- Передаем вес BLIP-2
            chunk_size=args.chunk,
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