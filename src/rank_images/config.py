# src/rank_images/config.py
"""
Конфигурационные константы для проекта rank_images.

Этот модуль определяет глобальные настройки, такие как доступность GPU,
используемый тип данных (dtype) и другие параметры, влияющие на работу моделей.
"""
import torch

# --- Устройства и типы данных ---
# Проверка наличия CUDA GPU.
USE_GPU: bool = torch.cuda.is_available()
"""
bool: Флаг, указывающий, доступна ли CUDA GPU для вычислений.
"""

# Определение типа данных в зависимости от наличия GPU.
# float16 используется для GPU для экономии памяти и ускорения вычислений.
# float32 используется для CPU.
DTYPE = torch.float16 if USE_GPU else torch.float32
"""
torch.dtype: Тип данных по умолчанию для тензоров.
             torch.float16 если доступен GPU, иначе torch.float32.
"""

# Определение устройств CPU и GPU.
DEVICE_CPU = torch.device("cpu")
"""
torch.device: Устройство CPU.
"""

DEVICE_GPU = torch.device("cuda") if USE_GPU else DEVICE_CPU
"""
torch.device: Устройство GPU (если доступно), иначе CPU.
"""

# --- Параметры моделей ---
# Максимальное количество токенов для обработки SigLIP за один раз.
# Это ограничение связано с архитектурой модели и объемом доступной памяти.
MAX_SIG_TOK: int = 64
"""
int: Максимальное количество токенов, обрабатываемых SigLIP-2 за один проход.
     Используется для разбиения длинных текстов на фрагменты.
"""

# --- Веса метрик по умолчанию ---
# Эти значения используются как стандартные веса при вычислении итогового балла.
# Они могут быть переопределены через CLI или JSON-конфигурацию.
ALPHA_DEFAULT: float = 0.6
"""
float: Вес метрики SigLIP (схожесть изображения и текста) в итоговом балле.
       По умолчанию 0.6.
"""

BETA_DEFAULT: float = 0.4
"""
float: Вес метрики Florence-2 (поиск объектов по запросу) в итоговом балле.
       По умолчанию 0.4.
"""

GAMMA_DEFAULT: float = 0.2
"""
float: Вес метрики CLIP-IQA (общее качество изображения) в итоговом балле.
       По умолчанию 0.2.
"""

DELTA_DEFAULT: float = 0.1
"""
float: Вес метрики DINOv2 (внутренние признаки изображения) в итоговом балле.
       По умолчанию 0.1.
"""

EPSILON_DEFAULT: float = 0.3
"""
float: Вес метрики BLIP-2 (Image-Text Matching) в итоговом балле.
       По умолчанию 0.3.
"""

ZETA_DEFAULT: float = 0.25
"""
float: Вес метрики BLIP Caption + BERTScore к prompt в итоговом балле.
       По умолчанию 0.25.
"""

# --- Имена моделей ---
# Константы для путей к предобученным моделям Hugging Face.
SIZE: str = "base" # "large"
PATCH: str = "16-224" # "large"

SIGLIP_MODEL_NAME: str = f"google/siglip2-{SIZE}-patch{PATCH}"
"""
str: Имя предобученной модели SigLIP-2 в Hugging Face Hub.
"""

DINO_MODEL_NAME: str = f"facebook/dinov2-{SIZE}"
"""
str: Имя предобученной модели DINOv2 в Hugging Face Hub.
"""

FLORENCE_MODEL_NAME: str = f"microsoft/Florence-2-{SIZE}"
"""
str: Имя предобученной модели Florence-2 в Hugging Face Hub.
"""

# BLIP2_SIZE: str = "base" # или "large"
BLIP2_SIZE: str = SIZE # или "large" (если нужно использовать большую версию)
# Используем условие для выбора
if BLIP2_SIZE == "large":
    BLIP2_ITM_MODEL_NAME: str = "Salesforce/blip2-itm-vit-L"
else: # base или по умолчанию
    BLIP2_ITM_MODEL_NAME: str = "Salesforce/blip2-itm-vit-g"


BLIP_CAPTION_MODEL_NAME: str = f"Salesforce/blip-image-captioning-{SIZE}"# <-- Имя модели BLIP Caption
"""
str: Имя предобученной модели BLIP для генерации описаний (captioning) в Hugging Face Hub.
"""

# --- Пути к папкам ---
# Путь к папке для оффлоада слоёв Florence-2, если они не помещаются в память.
FLORENCE_OFFLOAD_FOLDER: str = "florence_offload"
"""
str: Имя папки для оффлоада слоёв модели Florence-2 при нехватке памяти GPU.
"""