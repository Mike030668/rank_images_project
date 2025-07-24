# src/rank_images/models.py
"""
Загрузка и хранение моделей искусственного интеллекта.

Этот модуль отвечает за загрузку всех необходимых моделей (SigLIP-2, DINOv2,
Florence-2, CLIP-IQA) и предоставляет к ним доступ через глобальные переменные.
Логика загрузки учитывает доступность GPU и оптимизации памяти.
"""
import logging
from typing import TYPE_CHECKING

# Импорт библиотек для работы с моделями
import torch
from torchmetrics.multimodal import CLIPImageQualityAssessment
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoProcessor as FlorenceProcessor,
    AutoModelForCausalLM,
)

# Импорт конфигурационных констант
from .config import (
    SIGLIP_MODEL_NAME,
    DINO_MODEL_NAME,
    FLORENCE_MODEL_NAME,
    FLORENCE_OFFLOAD_FOLDER,
    DTYPE,
    DEVICE_CPU,
)

# Импорт утилит для работы с устройствами
from .device_utils import _to_gpu, _release # Импортируем для документации

# Настройка логгирования
logger = logging.getLogger(__name__)

# --- Глобальные переменные для моделей ---
# Они будут инициализированы при первом импорте этого модуля.

# --- CLIP-IQA ---
# Модель оценки качества изображения. Загружается на CPU и остаётся там.
iqa_metric: CLIPImageQualityAssessment = None
"""
CLIPImageQualityAssessment: Модель CLIP-IQA для оценки общего качества изображения.
                            Загружается один раз и всегда находится на CPU.
"""

# --- SigLIP-2 ---
# Процессор и модель SigLIP-2. Загружаются на CPU.
sig_proc: AutoProcessor = None
"""
AutoProcessor: Процессор для модели SigLIP-2. Подготавливает изображения и текст.
"""

sig_model: AutoModel = None
"""
AutoModel: Модель SigLIP-2 для оценки схожести изображения и текста.
           Загружается на CPU и временно перемещается на GPU во время инференса.
"""

# --- DINOv2 ---
# Процессор и модель DINOv2. Загружаются на CPU.
dino_proc: AutoProcessor = None
"""
AutoProcessor: Процессор для модели DINOv2. Подготавливает изображения.
"""

dino_model: AutoModel = None
"""
AutoModel: Модель DINOv2 для извлечения признаков изображения.
           Загружается на CPU и временно перемещается на GPU во время инференса.
"""

# --- Florence-2 ---
# Процессор и модель Florence-2. Загружаются лениво.
flor_proc: FlorenceProcessor = None
"""
AutoProcessor: Процессор для модели Florence-2. Подготавливает изображения и текст.
"""

flor_model: AutoModelForCausalLM = None
"""
AutoModelForCausalLM: Модель Florence-2 для задач генерации и grounding.
                      Загружается с `device_map="auto"` для оптимального распределения.
"""


def _load_florence(local_only: bool) -> AutoModelForCausalLM:
    """
    Внутренняя функция для загрузки модели Florence-2.

    Использует `device_map="auto"` для автоматического распределения слоёв,
    `torch_dtype=torch.float16` для экономии памяти (если GPU доступен),
    `low_cpu_mem_usage=True` для снижения потребления RAM при загрузке,
    и `offload_folder` для хранения слоёв, которые не помещаются в память.

    Args:
        local_only (bool): Если True, загружает модель только из локального кэша.
                           Если False, позволяет загрузить модель из интернета.

    Returns:
        AutoModelForCausalLM: Загруженная модель Florence-2.
    """
    return AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL_NAME,
        trust_remote_code=True,
        device_map="auto",  # Автоматическое распределение по устройствам
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,  # Экономия RAM при загрузке
        offload_folder=FLORENCE_OFFLOAD_FOLDER,  # Папка для оффлоада
        local_files_only=local_only,  # Только локальные файлы или разрешить загрузку
    )


def load_models() -> None:
    """
    Инициализирует и загружает все модели проекта.

    Эта функция должна быть вызвана один раз перед началом использования моделей.
    Она устанавливает значения глобальных переменных моделей.
    """
    global iqa_metric, sig_proc, sig_model, dino_proc, dino_model, flor_proc, flor_model

    logger.info("Начинаю загрузку моделей (на CPU)...")
    #print("[DEBUG_LOAD_MODELS] Начало выполнения load_models()") # <-- Добавлено
    # --- Загрузка CLIP-IQA ---
    iqa_metric = CLIPImageQualityAssessment("clip_iqa").to(DEVICE_CPU).eval()
    logger.info("Модель CLIP-IQA загружена.")

    # --- Загрузка SigLIP-2 ---
    sig_proc = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
    #print(f"[DEBUG_LOAD_MODELS] sig_proc присвоен: {type(sig_proc)}") # <-- Добавлено
    sig_model = AutoModel.from_pretrained(
        SIGLIP_MODEL_NAME, torch_dtype=DTYPE, device_map="cpu"
    ).eval()
    #print(f"[DEBUG_LOAD_MODELS] sig_model присвоен: {type(sig_model)}") # <-- Добавлено
    logger.info("Модель SigLIP-2 загружена.")

    # --- Загрузка DINOv2 ---
    dino_proc = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
    #print(f"[DEBUG_LOAD_MODELS] dino_proc присвоен: {type(dino_proc)}") # <-- Добавлено
    dino_model = AutoModel.from_pretrained(
        DINO_MODEL_NAME, torch_dtype=DTYPE, device_map="cpu"
    ).eval()
    #print(f"[DEBUG_LOAD_MODELS] dino_model присвоен: {type(dino_model)}") # <-- Добавлено
    logger.info("Модель DINOv2 загружена.")

    # --- Ленивая загрузка Florence-2 ---
    logger.info("Попытка загрузить Florence-2 из локального кэша...")
    try:
        # Сначала пытаемся загрузить из локального кэша для быстрого старта
        flor_model = _load_florence(local_only=True)
        logger.info("Florence-2 успешно загружена из локального кэша.")
    except (OSError, ValueError) as e:
        logger.info(f"Florence-2 не найдена локально ({e}), начинаю загрузку из интернета...")
        # Если локально не найдена, загружаем из интернета
        flor_model = _load_florence(local_only=False)
        logger.info("Florence-2 успешно загружена из интернета.")

    # Загрузчик процессора для Florence-2
    flor_proc = FlorenceProcessor.from_pretrained(FLORENCE_MODEL_NAME, trust_remote_code=True)

    logger.info("Все модели успешно загружены и готовы к использованию.")
    #print("[DEBUG_LOAD_MODELS] load_models() завершена.") # <-- Добавлено

# --- Инициализация моделей при импорте модуля (если это необходимо) ---
# В текущей логике, загрузка происходит в CLI или в ранжировании.
# Поэтому инициализация здесь не производится автоматически.
# load_models() # <- Эта строка будет раскомментирована, если нужно
#                #    автоматически загружать модели при импорте модуля.