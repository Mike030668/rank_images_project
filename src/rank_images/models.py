# src/rank_images/models.py
"""
Загрузка и хранение моделей искусственного интеллекта.

Этот модуль отвечает за загрузку всех необходимых моделей (SigLIP-2, DINOv2,
Florence-2, CLIP-IQA, BLIP-2 ITM, BLIP Caption) и предоставляет к ним доступ
через глобальные переменные. Логика загрузки учитывает доступность GPU
и оптимизации памяти.

Архитектура поддерживает гибкую загрузку:
- Карта METRIC_TO_MODELS определяет, какие глобальные переменные
  соответствуют каждой метрике.
- load_models() может принимать список включённых метрик и
  загружать только нужные модели.
"""
import sys
import logging
from typing import TYPE_CHECKING, List, Optional, Dict, Set
import torch

from torchmetrics.multimodal import CLIPImageQualityAssessment
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoProcessor as FlorenceProcessor,
    AutoModelForCausalLM,
    # --- BLIP-2 ---
    Blip2Processor,
    Blip2ForImageTextRetrieval, # (для ITM)
    Blip2ForConditionalGeneration, # (для captioning)
    # -----------------
    BlipProcessor,
    BlipForConditionalGeneration
)

# Импорт конфигурационных констант
from .config import (
    SIGLIP_MODEL_NAME,
    DINO_MODEL_NAME,
    FLORENCE_MODEL_NAME,
    BLIP2_ITM_MODEL_NAME, # <-- (для ITM)
    BLIP_CAPTION_MODEL_NAME, # (для captioning)
    BLIP2_CAPTION_MODEL_NAME, # (для captioning)
    # ------------------
    FLORENCE_OFFLOAD_FOLDER,
    DTYPE,
    DEVICE_CPU,
)

import ImageReward as RM


# --- ИМПОРТ ДЛЯ ПАЙПЛАЙНА ---
# Импортируем функцию для получения списка всех метрик
from .pipeline_config import get_all_metrics

# Импорт утилит для работы с устройствами
from .device_utils import _to_gpu, _release

if TYPE_CHECKING:
    # Импорты только для типизации, чтобы избежать циклических импортов
    from transformers import (
        AutoProcessor as SiglipProcessor,
        AutoModel as SiglipModel,
        AutoProcessor as DinoProcessor,
        AutoModel as DinoModel,
        AutoProcessor as FlorenceProcessorType,
        AutoModelForCausalLM as FlorenceModel,
        BlipProcessor as BlipCapProc,
        BlipForConditionalGeneration as BlipCapModel,
        Blip2Processor as Blip2Proc,
        Blip2ForImageTextRetrieval as Blip2ITRModel,
        Blip2ForConditionalGeneration as Blip2CapModel,
    )
    from torchmetrics.multimodal import CLIPImageQualityAssessment as IQAMetric
    

# Настройка логгирования
logger = logging.getLogger(__name__)

# --- Глобальные переменные для моделей ---
# Они будут инициализированы при первом импорте этого модуля
# и/или при вызове load_models().

# --- CLIP-IQA ---
iqa_metric: Optional['IQAMetric'] = None
"""
CLIPImageQualityAssessment | None: Модель CLIP-IQA для оценки общего качества изображения.
                                    Загружается один раз и всегда находится на CPU.
"""

# --- SigLIP-2 ---
sig_proc: Optional['SiglipProcessor'] = None
"""
AutoProcessor | None: Процессор для модели SigLIP-2. Подготавливает изображения и текст.
"""

sig_model: Optional['SiglipModel'] = None
"""
AutoModel | None: Модель SigLIP-2 для оценки схожести изображения и текста.
                  Загружается на CPU и временно перемещается на GPU во время инференса.
"""

# --- DINOv2 ---
dino_proc: Optional['DinoProcessor'] = None
"""
AutoProcessor | None: Процессор для модели DINOv2. Подготавливает изображения.
"""

dino_model: Optional['DinoModel'] = None
"""
AutoModel | None: Модель DINOv2 для извлечения признаков изображения.
                  Загружается на CPU и временно перемещается на GPU во время инференса.
"""

# --- Florence-2 ---
flor_proc: Optional['FlorenceProcessorType'] = None
"""
AutoProcessor | None: Процессор для модели Florence-2. Подготавливает изображения и текст.
"""

flor_model: Optional['FlorenceModel'] = None
"""
AutoModelForCausalLM | None: Модель Florence-2 для задач генерации и grounding.
                             Загружается с `device_map="auto"` для оптимального распределения.
"""

# --- BLIP-2 ---
blip2_processor: Optional['Blip2Proc'] = None
"""
Blip2Processor | None: Процессор для модели BLIP-2 ITM.
"""

blip2_model: Optional['Blip2ITRModel'] = None
"""
Blip2ForConditionalGeneration | None: Модель BLIP-2 для Image-Text Matching.
                                      Загружается на CPU и временно перемещается на GPU.
"""
# -------------

# --- BLIP Caption ---
blip_cap_processor: Optional['BlipCapProc'] = None
"""
BlipProcessor | None: Процессор для модели BLIP Caption.
"""

blip_cap_model: Optional['BlipCapModel'] = None
"""
BlipForConditionalGeneration | None: Модель BLIP для генерации описаний (captioning).
                                     Загружается на CPU и временно перемещается на GPU.
"""
# --- BLIP-2 Caption ---
blip2_cap_processor:  Optional['Blip2Proc'] = None
"""
Blip2Processor: Процессор для модели BLIP-2 Caption (ConditionalGeneration).
"""

blip2_cap_model: Optional['Blip2CapModel'] = None
"""
Blip2ForConditionalGeneration: Модель BLIP-2 для генерации описаний (captioning).
                              Используется для blip_2_caption_bertscore.
"""


imr_model  = None        # reward-модель
"""
«человеческая» оценка стиля/композиции.
"""

# --------------------

# --- Карта соответствия метрик и моделей ---
# Это центральное место, определяющее, какие глобальные переменные
# необходимы для каждой метрики. Используется load_models() и другими модулями.
METRIC_TO_MODELS: Dict[str, List[str]] = {
    "sig": ["sig_proc", "sig_model"],
    "flor": ["flor_proc", "flor_model"],
    "iqa": ["iqa_metric"],
    "dino": ["dino_proc", "dino_model"],
    "blip2": ["blip2_processor", "blip2_model"],
    "blip_cap": ["blip_cap_processor", "blip_cap_model"],
    # Для blip2_caption_bertscore (новая метрика) потребуется blip2_caption_model
    "blip2_cap": ["blip2_cap_processor", "blip2_cap_model"], 
    "imr": ["imr_model"], 
}
"""
Dict[str, List[str]]: Карта соответствия между именами метрик и
                      списками имён глобальных переменных моделей/процессоров,
                      необходимых для их работы.

                      Это центральный источник истины для определения
                      зависимостей метрик от моделей.
"""
# --------------------------------------------

# --- Вспомогательные функции для загрузки моделей ---
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

# --- Основная функция загрузки моделей ---
# src/rank_images/models.py
def load_models(enabled_metrics_list: Optional[List[str]] = None) -> None:
    """
    Инициализирует и загружает модели проекта.

    Эта функция должна быть вызвана один раз перед началом использования моделей.
    Она устанавливает значения глобальных переменных моделей.

    Args:
        enabled_metrics_list (List[str] | None): Список включённых метрик.
            Если None, загружаются все модели.
    """
    global iqa_metric, sig_proc, sig_model, dino_proc, dino_model, flor_proc, flor_model
    global blip2_processor, blip2_model
    global blip_cap_processor, blip_cap_model
    # --- ДОБАВИТЬ НОВЫЕ ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
    global blip2_cap_processor, blip2_cap_model
    global imr_model
    # --------------------------------------------

    logger.info("Начинаю загрузку моделей (на CPU)...")
    # --- ОПРЕДЕЛЕНИЕ, КАКИЕ МОДЕЛИ НУЖНО ЗАГРУЖАТЬ ---
    models_to_load: Set[str] = set()
    if enabled_metrics_list is None:
        # Если список не передан, загружаем все модели
        logger.debug("Список метрик не указан. Загружаю все модели.")
        # Используем центральную карту METRIC_TO_MODELS
        models_to_load.update(sum(METRIC_TO_MODELS.values(), []))
    elif not enabled_metrics_list:
        # Если передан пустой список, ничего не загружаем
        logger.info("Список метрик пуст. Никакие модели не будут загружены.")
        return
    else:
        # Если передан список, загружаем модели только для включённых метрик
        logger.debug(f"Загружаю модели только для метрик: {enabled_metrics_list}")
        for metric_name in enabled_metrics_list:
            # Получаем список моделей для метрики из центральной карты
            required_models = METRIC_TO_MODELS.get(metric_name, [])
            models_to_load.update(required_models)
            logger.debug(f"  Метрика '{metric_name}' требует модели: {required_models}")
    
    logger.debug(f"Итоговый список моделей для загрузки: {sorted(models_to_load)}")
    # ------------------------------------------

    # --- Загрузка моделей ---
    try:
        # Загрузка CLIP-IQA
        if "iqa_metric" in models_to_load:
            logger.info("Начинаю загрузку модели CLIP-IQA...")
            iqa_metric = CLIPImageQualityAssessment("clip_iqa").to(DEVICE_CPU).eval()
            logger.info("Модель CLIP-IQA загружена.")
        else:
            iqa_metric = None
            logger.info("Модель CLIP-IQA пропущена (не включена).")
        # --- Конец загрузки CLIP-IQA --

        # Загрузка SigLIP-2
        if "sig_proc" in models_to_load and "sig_model" in models_to_load:
            logger.info("Начинаю загрузку модели SigLIP-2...")
            sig_proc = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            sig_model = AutoModel.from_pretrained(
                SIGLIP_MODEL_NAME, torch_dtype=DTYPE, device_map="cpu"
            ).eval()
            logger.info("Модель SigLIP-2 загружена.")
        else:
            sig_proc, sig_model = None, None
            logger.info("Модель SigLIP-2 пропущена (не включена).")
        # --- Конец загрузки SigLIP-2 ---
        
        # Загрузка DINOv2
        if "dino_proc" in models_to_load and "dino_model" in models_to_load:
            logger.info("Начинаю загрузку модели DINOv2...")
            dino_proc = AutoProcessor.from_pretrained(DINO_MODEL_NAME)
            dino_model = AutoModel.from_pretrained(
                DINO_MODEL_NAME, torch_dtype=DTYPE, device_map="cpu"
            ).eval()
            logger.info("Модель DINOv2 загружена.")
        else:
            dino_proc, dino_model = None, None
            logger.info("Модель DINOv2 пропущена (не включена).")
        # --- Конец загрузки DINOv2 ---

        # Загрузка Florence-2
        if "flor_proc" in models_to_load and "flor_model" in models_to_load:
            logger.info("Попытка загрузить Florence-2 из локального кэша...")
            try:
                flor_model = _load_florence(local_only=True)
                logger.info("Florence-2 успешно загружена из локального кэша.")
            except Exception as e:
                logger.error(f"Florence-2 не найдена локально ({e}), начинаю загрузку из интернета...")
                try:
                    flor_model = _load_florence(local_only=False)
                    logger.info("Florence-2 успешно загружена из интернета.")
                except Exception as e_internet:
                    logger.error(f"Ошибка при загрузке Florence-2 из интернета: {e_internet}", exc_info=True)
                    flor_model = None
                    logger.warning("Модель Florence-2 не будет доступна.")
            
            if flor_model is not None:
                flor_proc = FlorenceProcessor.from_pretrained(FLORENCE_MODEL_NAME, trust_remote_code=True)
                logger.info("Процессор Florence-2 загружен.")
            # --- Конец загрузки Florence-2 ---


        # Загрузка BLIP-2 ITM
        if "blip2_processor" in models_to_load and "blip2_model" in models_to_load:
            logger.info("Начинаю загрузку модели BLIP-2...")
            blip2_processor = Blip2Processor.from_pretrained(BLIP2_ITM_MODEL_NAME)
            blip2_model = Blip2ForImageTextRetrieval.from_pretrained(
                BLIP2_ITM_MODEL_NAME,
                torch_dtype=DTYPE,
                device_map="cpu",
                low_cpu_mem_usage=True, # <-- Добавить
            ).eval()
            logger.info("Модель BLIP-2 (ITM) загружена.")
        else:
            blip2_processor, blip2_model = None, None
            logger.info("Модель BLIP-2 (ITM) пропущена (не включена).")
        # --- Конец загрузки BLIP-2 ---

        # Загрузка BLIP Caption
        if "blip_cap_processor" in models_to_load and "blip_cap_model" in models_to_load:
            logger.info("Начинаю загрузку модели BLIP Caption...")
            blip_cap_processor = BlipProcessor.from_pretrained(BLIP_CAPTION_MODEL_NAME)
            blip_cap_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_CAPTION_MODEL_NAME,
                torch_dtype=DTYPE,
                device_map="cpu"
            ).eval()
            logger.info("Модель BLIP Caption загружена.")
        else:
            blip_cap_processor, blip_cap_model = None, None
            logger.info("Модель BLIP Caption пропущена (не включена).")
        # --- Конец загрузки BLIP Caption ---

        # --- Загрузка BLIP-2 Caption ---
        if "blip2_cap_processor" in models_to_load and "blip2_cap_model" in models_to_load:
            logger.info("Начинаю загрузку модели BLIP-2 Caption...")
            try:
                # BLIP-2 Caption будет загружаться на CPU и перемещаться на GPU во время инференса
                blip2_cap_processor = Blip2Processor.from_pretrained(BLIP2_CAPTION_MODEL_NAME) # <-- Используем BLIP2Processor
                blip2_cap_model = Blip2ForConditionalGeneration.from_pretrained( # <-- Используем Blip2ForConditionalGeneration
                    BLIP2_CAPTION_MODEL_NAME, # <-- Имя модели из config.py
                    torch_dtype=DTYPE,
                    device_map="cpu"
                ).eval()
                logger.info("Модель BLIP-2 Caption загружена.")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели BLIP-2 Caption: {e}", exc_info=True)
                # Продолжаем работу без BLIP-2 Caption
                blip2_cap_processor = None
                blip2_cap_model = None
                logger.warning("Модель BLIP-2 Caption не будет доступна.")
        else:
            blip2_cap_processor, blip2_cap_model = None, None
            logger.info("Модель BLIP-2 Caption пропущена (не включена).")
        # --- Конец загрузки BLIP-2 Caption ---

        #--- Загрузка ImageReward ---
        if "imr" in enabled_metrics_list and imr_model is None:
            imr_model = RM.load("ImageReward-v1.0")
            imr_model.device = torch.device("cpu")   # важный переключатель
            imr_model.to("cpu")
            #--- Конец загрузки ImageReward ---
        logger.info("Загрузка моделей завершена.")

    except Exception as e:
        logger.error(f"Ошибка при загрузке моделей", exc_info=True)
        logger.warning("Загрузка моделей не завершена.")
    # --- Конец загрузки BLIP Caption ---
# --- Конец модуля ---