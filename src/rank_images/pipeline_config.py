# src/rank_images/pipeline_config.py
"""
Конфигурация пайплайна ранжирования.

Этот модуль предоставляет функции для загрузки, валидации и предоставления
конфигурации пайплайна ранжирования из JSON-файла. Он определяет, какие метрики
включены, их веса по умолчанию и другие параметры обработки.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from .config import ( # Импортируем значения по умолчанию
    ALPHA_DEFAULT,
    BETA_DEFAULT,
    GAMMA_DEFAULT,
    DELTA_DEFAULT,
    EPSILON_DEFAULT,
    ZETA_DEFAULT, 
    THETA_DEFAULT,
    ALL_METRICS,
    # Другие константы по необходимости
)
from .models import METRIC_TO_MODELS # <-- Импорт для документации/проверки

logger = logging.getLogger(__name__)

# --- Стандартная конфигурация (если файл не указан или повреждён) ---
DEFAULT_PIPELINE_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "enabled_metrics": ALL_METRICS,
        "default_weights": {
            "alpha": ALPHA_DEFAULT,
            "beta": BETA_DEFAULT,
            "gamma": GAMMA_DEFAULT,
            "delta": DELTA_DEFAULT,
            "epsilon": EPSILON_DEFAULT,
            "zeta": ZETA_DEFAULT,
                        # --- НОВОЕ ---
            "theta": THETA_DEFAULT, # <-- НОВОЕ: вес для blip2_cap
        }
    },
    "processing": {
        "chunk_size": None # Использовать значение из config.py по умолчанию
    }
}
"""
dict: Стандартная конфигурация пайплайна по умолчанию.
      Используется, если файл конфигурации не предоставлен или недействителен.
"""

def load_pipeline_config(config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """
    Загружает и валидирует конфигурацию пайплайна из JSON-файла.

    Args:
        config_path (str | Path | None): Путь к JSON-файлу конфигурации.
                                         Если None, возвращается стандартная конфигурация.

    Returns:
        dict: Словарь с конфигурацией пайплайна.
              Включает ключи 'pipeline' и 'processing'.
    """
    if config_path is None:
        logger.info("Путь к конфигурации пайплайна не указан. Использую стандартную конфигурацию.")
        return DEFAULT_PIPELINE_CONFIG.copy() # Возвращаем копию по умолчанию

    config_file = Path(config_path)
    if not config_file.exists():
        logger.warning(f"Файл конфигурации пайплайна '{config_file}' не найден. Использую стандартную конфигурацию.")
        return DEFAULT_PIPELINE_CONFIG.copy()

    logger.info(f"Загружаю конфигурацию пайплайна из: {config_file}")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = json.load(f)

         
        # --- ВРЕМЕННЫЙ ОТЛАДОЧНЫЙ ПРИНТ ---
        print(f"[DEBUG_PIPELINE_CONFIG] Загруженный config: {user_config}")
        print(f"[DEBUG_PIPELINE_CONFIG] type(config): {type(user_config)}")
        print(f"[DEBUG_PIPELINE_CONFIG] config.get('pipeline'): {user_config.get('pipeline')}")
        print(f"[DEBUG_PIPELINE_CONFIG] type(config.get('pipeline')): {type(user_config.get('pipeline'))}")
        print(f"[DEBUG_PIPELINE_CONFIG] config.get('pipeline', {{}}).get('enabled_metrics'): {user_config.get('pipeline', {}).get('enabled_metrics')}")
        print(f"[DEBUG_PIPELINE_CONFIG] type(config.get('pipeline', {{}}).get('enabled_metrics')): {type(user_config.get('pipeline', {}).get('enabled_metrics'))}")
        # ----------------------------------           
        
        # --- Базовая валидация ---
        if not isinstance(user_config, dict):
            raise ValueError("Конфигурационный файл должен содержать JSON-объект (словарь).")
        
        # Убедимся, что ключи верхнего уровня существуют
        user_config.setdefault("pipeline", {})
        user_config.setdefault("processing", {})

        # --- Валидация и объединение pipeline.enabled_metrics ---
        pipeline_section = user_config["pipeline"]
        if "enabled_metrics" not in pipeline_section:
            logger.info("Ключ 'enabled_metrics' не найден в секции 'pipeline'. Использую все метрики по умолчанию.")
            pipeline_section["enabled_metrics"] = DEFAULT_PIPELINE_CONFIG["pipeline"]["enabled_metrics"]
        else:
            # Проверяем, что это список
            if not isinstance(pipeline_section["enabled_metrics"], list):
                 logger.warning("Ключ 'enabled_metrics' в секции 'pipeline' должен быть списком. Использую все метрики по умолчанию.")
                 pipeline_section["enabled_metrics"] = DEFAULT_PIPELINE_CONFIG["pipeline"]["enabled_metrics"]
            else:
                # Проверяем, что все элементы - строки
                valid_metrics = [m for m in pipeline_section["enabled_metrics"] if isinstance(m, str)]
                if len(valid_metrics) != len(pipeline_section["enabled_metrics"]):
                    logger.warning("Некоторые элементы в 'enabled_metrics' не являются строками. Игнорирую их.")
                    pipeline_section["enabled_metrics"] = valid_metrics
                
        # --- Валидация и объединение pipeline.default_weights ---
        weights_section = pipeline_section.setdefault("default_weights", {})
        default_weights = DEFAULT_PIPELINE_CONFIG["pipeline"]["default_weights"]
        for weight_name, default_value in default_weights.items():
            if weight_name not in weights_section:
                logger.debug(f"Вес '{weight_name}' не найден в конфигурации. Использую значение по умолчанию: {default_value}.")
                weights_section[weight_name] = default_value
            # Можно добавить проверку типа, если нужно
        
        # --- Валидация processing.chunk_size ---
        processing_section = user_config["processing"]
        # chunk_size может быть None или int, ничего не делаем, оставляем как есть
        
        logger.info("Конфигурация пайплайна успешно загружена и проверена.")
        return user_config

    # --- Обработка ошибок ---
    except json.JSONDecodeError as e:
        # logger.error(f"Ошибка декодирования JSON в файле '{config_file}': {e}") # <-- Старая строка
        # --- ОБНОВЛЕНО ---
        logger.error(f"Ошибка декодирования JSON в файле '{config_file}': {e}", exc_info=True) # <-- НОВАЯ строка
        # ---------------
        logger.info("Использую стандартную конфигурацию.")
        return DEFAULT_PIPELINE_CONFIG.copy()
    except Exception as e:
        # logger.error(f"Неожиданная ошибка при загрузке конфигурации из '{config_file}': {e}") # <-- Старая строка
        # --- ОБНОВЛЕНО ---
        logger.error(f"Неожиданная ошибка при загрузке конфигурации из '{config_file}': {e}", exc_info=True) # <-- НОВАЯ строка
        # ---------------
        logger.info("Использую стандартную конфигурацию.")
        return DEFAULT_PIPELINE_CONFIG.copy()
    # --- Конец обработки ошибок ---

# --- Вспомогательные функции для удобства доступа ---
# src/rank_images/pipeline_config.py
# --- Вспомогательные функции ---
def get_all_metrics(config: Dict[str, Any]) -> List[str]:
    """Извлекает список ВСЕХ доступных метрик из конфигурации."""
    # Возвращаем центральный список из config.py
    # Это гарантирует, что список всегда актуален
    # return ALL_METRICS.copy() # Возвращаем копию, чтобы избежать мутаций
    # --- ОБНОВЛЕНО ---
    # Используем METRIC_TO_MODELS из models.py для получения списка
    from .models import METRIC_TO_MODELS
    return list(METRIC_TO_MODELS.keys())
    # ---------------
# --- Конец вспомогательных функций ---

def get_enabled_metrics(config: Dict[str, Any]) -> List[str]:
    """
    Извлекает список включённых метрик из конфигурации пайплайна.

    Args:
        config (dict): Конфигурация пайплайна, загруженная load_pipeline_config().

    Returns:
        List[str]: Список имён включённых метрик. Если конфигурация
                   повреждена или метрики не указаны, возвращается
                   стандартный список из DEFAULT_PIPELINE_CONFIG.
    """
    try:
        # Извлекаем список из конфига
        # config["pipeline"]["enabled_metrics"]
        enabled_metrics_from_config = config.get("pipeline", {}).get("enabled_metrics", [])
        
        # Проверяем, что это список
        if not isinstance(enabled_metrics_from_config, list):
            logger.warning(
                f"'enabled_metrics' в конфиге должен быть списком. "
                f"Получен тип: {type(enabled_metrics_from_config)}. "
                f"Использую стандартный список."
            )
            # Возвращаем стандартный список из DEFAULT_PIPELINE_CONFIG
            return DEFAULT_PIPELINE_CONFIG["pipeline"]["enabled_metrics"]
        
        # Проверяем, что все элементы - строки
        valid_metrics = [m for m in enabled_metrics_from_config if isinstance(m, str)]
        if len(valid_metrics) != len(enabled_metrics_from_config):
            logger.warning(
                f"Некоторые элементы в 'enabled_metrics' не являются строками. "
                f"Игнорирую их. Валидные: {valid_metrics}"
            )
            
        logger.debug(f"Извлечённые включённые метрики: {valid_metrics}")
        return valid_metrics

    except Exception as e:
        logger.error(f"Ошибка при извлечении 'enabled_metrics' из конфига: {e}", exc_info=True)
        # Возвращаем стандартный список
        return DEFAULT_PIPELINE_CONFIG["pipeline"]["enabled_metrics"]

# src/rank_images/pipeline_config.py
def get_default_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """Извлекает словарь весов по умолчанию из конфигурации."""
    return config.get("pipeline", {}).get("default_weights", {})

# src/rank_images/pipeline_config.py
def get_chunk_size(config: Dict[str, Any]) -> Optional[int]:
    """Извлекает размер чанка из конфигурации."""
    return config.get("processing", {}).get("chunk_size", None)

# --- Конец модуля ---