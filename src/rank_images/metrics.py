# src/rank_images/metrics.py
"""
Функции для вычисления метрик качества изображений.

Этот модуль содержит функции, каждая из которых реализует отдельную метрику
для оценки изображений. Это центральный модуль для расширения функциональности:
для добавления новой метрики нужно создать новую функцию здесь и затем
интегрировать её в логику ранжирования в `ranking.py`.

Все функции метрик должны быть снабжены декоратором `@torch.inference_mode()`
для оптимизации памяти и скорости во время инференса.

Примечание:
    Функции, использующие модели SigLIP-2 и DINOv2, должны использовать
    `_to_gpu` и `_release` из `device_utils` для временного перемещения
    модели на GPU и последующего освобождения памяти.
"""
import logging
from typing import List, Optional
import numpy as np
import torch
import torchvision
from PIL import Image

# Настройка логгирования
logger = logging.getLogger(__name__)
# --- ИМПОРТ BERTSCORE ---
try:
    from bert_score import score as bert_score_func
    BERT_SCORE_AVAILABLE = True
except ImportError:
    bert_score_func = None
    BERT_SCORE_AVAILABLE = False
    logger.warning(
        "Библиотека `bert_score` не найдена. Метрика BLIP Caption + BERTScore будет недоступна. "
        "Установите её с помощью `pip install bert_score`."
    )
# -----------------------

# Импорт утилит и моделей
from . import models
from .device_utils import _to_gpu, _release
from .config import MAX_SIG_TOK, DTYPE, DEVICE_CPU


# --- Вспомогательная функция нормализации ---
def _z(a: np.ndarray) -> np.ndarray:
    """
    Нормализует массив по Z-оценке (стандартная нормализация).

    Вычитает среднее значение массива и делит на стандартное отклонение.
    Если стандартное отклонение равно 0, возвращает массив нулей той же формы.

    Args:
        a (np.ndarray): Входной одномерный массив NumPy.

    Returns:
        np.ndarray: Нормализованный массив той же формы.
    """
    std_dev = a.std()
    if std_dev > 0:
        # Нормализация: (x - mean) / std
        return (a - a.mean()) / std_dev
    else:
        # Если std == 0, все значения одинаковы, возвращаем массив нулей
        return np.zeros_like(a)


# --- Метрики ---
# Каждая метрика принимает изображение PIL.Image.Image и, возможно,
# дополнительные аргументы (например, текстовые промпты).
# Возвращает вещественное число (float) как оценку.

#print(f"[DEBUG_IMPORT] В metrics.py после импортов: sig_proc is None = {models.sig_proc is None}")

@torch.inference_mode()
def get_siglip_score(img: Image.Image, txts: List[str]) -> float:
    """
    Вычисляет среднюю косинусную схожесть между изображением и списком текстов
    с помощью модели SigLIP-2.

    Для каждого текста в списке `txts` вычисляется схожесть с изображением.
    Возвращается среднее значение этих схожестей.

    Args:
        img (PIL.Image.Image): Входное изображение.
        txts (List[str]): Список текстовых запросов.

    Returns:
        float: Средняя косинусная схожесть. Если `txts` пуст, возвращает 0.0.
    """
    #print(f"[DEBUG_CALL] В get_siglip_score: sig_proc is None = {sig_proc is None}")

    if models.sig_proc is None:
        raise RuntimeError(
            "models.sig_proc равен None в get_siglip_score! "
            "Убедитесь, что load_models() была вызвана до ранжирования."
        )

    if not txts:
        logger.debug("Список текстов для SigLIP пуст. Возвращаю 0.0.")
        return 0.0

    logger.debug(f"Вычисляю SigLIP-2 скор для {len(txts)} текстов.")

    # 1. Подготовка данных (токенизация, преобразование изображения)
    #    с разбиением текста на фрагменты по MAX_SIG_TOK токенов.
    #    `sig_proc` уже знает о MAX_SIG_TOK, но мы явно указываем для контроля.
    feats = models.sig_proc( # <-- Обращение через models.
        images=img,
        text=txts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SIG_TOK,
    )

    # 2. Преобразование типа данных для входных тензоров (кроме input_ids)
    feats = {
        k: v.to(dtype=DTYPE) if k != "input_ids" else v for k, v in feats.items()
    }

    # 3. Перемещение модели на GPU (если доступен) и данных на то же устройство
    model_gpu = _to_gpu(models.sig_model) # <-- Обращение через models.
    feats = {k: v.to(model_gpu.device) for k, v in feats.items()}

    # 4. Прямой проход через модель
    try:
        out = model_gpu(**feats)
    finally:
        # 5. Всегда освобождаем модель после использования
        _release(model_gpu)

    # 6. Нормализация эмбеддингов
    im_embeds = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    txt_embeds = torch.nn.functional.normalize(out.text_embeds, dim=-1)

    # 7. Вычисление косинусной схожести и усреднение
    #    (im_embeds @ txt_embeds.T) дает матрицу схожестей [1 x N_txts]
    similarities = (im_embeds @ txt_embeds.T).squeeze(0) # [N_txts]
    mean_similarity = similarities.mean().item()

    logger.debug(f"Средняя SigLIP-2 схожесть: {mean_similarity:.4f}")
    return mean_similarity


@torch.inference_mode()
def get_florence_score(img: Image.Image, phrase: str) -> float:
    """
    Вычисляет "качество grounding" фразы на изображении с помощью Florence-2.

    Использует задачу `<CAPTION_TO_PHRASE_GROUNDING>` для поиска объектов,
    описанных в `phrase`. Возвращает отношение количества найденных объектов
    к количеству слов в фразе.

    Примечание:
        Модель Florence-2 НЕ перемещается `_to_gpu`, так как она уже
        оптимально размещена при загрузке (`device_map="auto"`).

    Args:
        img (PIL.Image.Image): Входное изображение.
        phrase (str): Текстовая фраза для поиска объектов.

    Returns:
        float: Отношение найденных объектов к количеству слов в фразе.
               Если фраза пуста, возвращает 0.0.
    """
    phrase = phrase.strip()
    if not phrase:
        logger.debug("Фраза для Florence пуста. Возвращаю 0.0.")
        return 0.0

    logger.debug(f"Вычисляю Florence-2 скор для фразы: '{phrase[:50]}...'")

    # 1. Определение задачи
    task = "<CAPTION_TO_PHRASE_GROUNDING>"

    # 2. Подготовка входных данных
    inputs = models.flor_proc(text=task + phrase, images=img, return_tensors="pt")

    # 3. Перемещение данных на устройство, где находится модель
    #    Определяем устройство и тип данных первого параметра модели
    first_param = next(models.flor_model.parameters())
    target_device = first_param.device
    model_dtype = first_param.dtype

    #    Перемещаем тензоры на нужное устройство
    #    pixel_values часто имеют другой тип (float), поэтому обрабатываем отдельно
    inputs_moved = {}
    for k, v in inputs.items():
        if k == "pixel_values":
            inputs_moved[k] = v.to(target_device, dtype=model_dtype)
        else:
            inputs_moved[k] = v.to(target_device)

    # 4. Генерация (инференс)
    generated_ids = models.flor_model.generate(
        input_ids=inputs_moved["input_ids"],
        pixel_values=inputs_moved["pixel_values"],
        max_new_tokens=512,
        do_sample=False,
        num_beams=3,
    )

    # 5. Декодирование результата
    generated_text = models.flor_proc.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # 6. Пост-обработка результата
    parsed_output = models.flor_proc.post_process_generation(
        generated_text, task=task, image_size=(img.width, img.height)
    )

    # 7. Извлечение найденных меток (labels)
    labels_found = parsed_output.get(task, {}).get("labels", [])
    num_labels_found = len(labels_found)

    # 8. Подсчет количества слов в фразе
    #    Заменяем запятые на пробелы и разбиваем на слова, отфильтровывая пустые
    words_in_phrase = [w for w in phrase.replace(",", " ").split() if w]
    num_words = len(words_in_phrase)

    # 9. Вычисление финального балла
    score = num_labels_found / max(num_words, 1) # Избегаем деления на 0

    logger.debug(
        f"Florence-2: найдено {num_labels_found} объектов из {num_words} слов. "
        f"Скор: {score:.4f}"
    )
    return score


@torch.inference_mode()
def get_iqa(img: Image.Image) -> float:
    """
    Оценивает общее качество изображения с помощью модели CLIP-IQA.

    Модель всегда находится на CPU.

    Args:
        img (PIL.Image.Image): Входное изображение.

    Returns:
        float: Оценка качества изображения.
    """
    logger.debug("Вычисляю CLIP-IQA скор.")

    # 1. Преобразование PIL Image в тензор PyTorch и добавление batch dimension
    #    Преобразуем в float32, так как iqa_metric ожидает этот тип на CPU
    img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    img_tensor = img_tensor.to(DEVICE_CPU, dtype=torch.float32)

    # 2. Прямой проход через модель (на CPU)
    quality_score = models.iqa_metric(img_tensor).item()

    logger.debug(f"CLIP-IQA скор: {quality_score:.4f}")
    return quality_score


@torch.inference_mode()
def get_dino(img: Image.Image) -> float:
    """
    Вычисляет L2-норму вектора признаков CLS-токена из модели DINOv2.

    Это косвенная метрика "внутреннего качества" или "структурированности"
    изображения.

    Args:
        img (PIL.Image.Image): Входное изображение.

    Returns:
        float: L2-норма вектора признаков CLS-токена.
    """
    logger.debug("Вычисляю DINOv2 скор.")

    # 1. Подготовка данных (преобразование изображения)
    feats = models.dino_proc(images=img, return_tensors="pt")

    # 2. Перемещение модели на GPU (если доступен) и данных на то же устройство
    model_gpu = _to_gpu(models.dino_model)
    feats = {k: v.to(model_gpu.device, dtype=DTYPE) for k, v in feats.items()}

    # 3. Прямой проход через модель
    try:
        output = model_gpu(**feats)
        # 4. Извлечение CLS-токена (первый токен в последовательности)
        cls_token_features = output.last_hidden_state[:, 0, :] # [1, D]
    finally:
        # 5. Всегда освобождаем модель после использования
        _release(model_gpu)

    # 6. Вычисление L2-нормы вектора признаков
    l2_norm = torch.linalg.vector_norm(cls_token_features, ord=2, dim=-1).item()

    logger.debug(f"DINOv2 L2-норма CLS-токена: {l2_norm:.4f}")
    return l2_norm


# --- НОВАЯ МЕТРИКА: BLIP-2 Matching Score ---
@torch.inference_mode()
def get_blip2_match_score(img: Image.Image, prompts: List[str]) -> float:
    """
    Вычисляет среднюю вероятность соответствия (Image-Text Matching) между
    изображением и списком текстовых промптов с помощью модели BLIP-2 ITM.

    Обрабатывает каждый промпт отдельно.

    Args:
        img (PIL.Image.Image): Входное изображение.
        prompts (List[str]): Список текстовых промптов для проверки на соответствие.

    Returns:
        float: Средняя вероятность соответствия. Если список `prompts` пуст
               или модель BLIP-2 не загружена, возвращает 0.0.
    """
    # Используем models.blip2_model и models.blip2_processor
    if models.blip2_processor is None or models.blip2_model is None: 
        logger.warning(
            "Модель или процессор BLIP-2 (ITM) не загружены. "
            "Возвращаю 0.0 для BLIP-2 скор."
        )
        return 0.0

    if not prompts:
        logger.debug("Список промптов для BLIP-2 пуст. Возвращаю 0.0.")
        return 0.0

    logger.debug(f"Вычисляю BLIP-2 ITM скор для {len(prompts)} промптов.")

    individual_scores = []
    model_gpu = None

    try:
        # Перемещаем модель на GPU (если доступен) один раз
        model_gpu = _to_gpu(models.blip2_model)

        for prompt in prompts:
            try:
                # 1. Подготовка входных данных для ОДНОГО текста
                inputs = models.blip2_processor(
                    images=img,
                    text=prompt, # Один текст
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                # 2. Преобразование типа данных
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=DTYPE)

                # 3. Перемещение данных на устройство модели
                inputs_moved = {k: v.to(model_gpu.device) for k, v in inputs.items()}

                # 4. Прямой проход через модель ITM с флагом
                # Используем logits_per_image, как в официальном примере
                outputs = model_gpu(
                    **inputs_moved,
                    use_image_text_matching_head=True
                )

                # 5. Извлечение и обработка логитов
                # Согласно примеру, используем logits_per_image
                logits_per_image = outputs.logits_per_image # Ожидаемая форма: [1, N_classes]
                logger.debug(f"  Промпт '{prompt[:20]}...': logits_per_image.shape = {logits_per_image.shape}")

                if logits_per_image.dim() != 2 or logits_per_image.shape[0] != 1:
                    logger.warning(f"Неожиданная форма logits_per_image для промпта '{prompt}': {logits_per_image.shape}")
                    continue

                num_classes = logits_per_image.shape[1]
                if num_classes < 1:
                    logger.warning(f"Недостаточно классов в logits_per_image для промпта '{prompt}': {logits_per_image.shape}")
                    continue

                # Применяем softmax для получения вероятностей
                # Согласно примеру, softmax по dim=1
                probs = torch.nn.functional.softmax(logits_per_image, dim=1)

                # 6. Извлечение вероятности "yes"
                # probs.shape [1, N_classes]
                if num_classes >= 2:
                    # probs[0, 1] - вероятность "yes" для первого (и единственного) изображения
                    match_prob = probs[0, 1].item()
                else:
                    # Если только один класс, предполагаем, что это и есть "yes"
                    # или вероятность соответствия напрямую
                    match_prob = probs[0, -1].item()

                individual_scores.append(match_prob)
                logger.debug(f"  Промпт '{prompt[:20]}...': вероятность соответствия = {match_prob:.4f}")

            except Exception as prompt_e:
                logger.error(f"Ошибка при обработке промпта '{prompt}': {prompt_e}")
                continue

        # 7. Вычисление среднего значения
        if individual_scores:
            average_match_prob = sum(individual_scores) / len(individual_scores)
            logger.debug(f"Средняя вероятность соответствия BLIP-2 (ITM) по {len(individual_scores)} промптам: {average_match_prob:.4f}")
            return average_match_prob
        else:
            logger.warning("Не удалось получить ни один скор для BLIP-2 ITM.")
            return 0.0

    except Exception as e:
        logger.error(f"Фатальная ошибка при вычислении BLIP-2 скор: {e}", exc_info=True)
        return 0.0
    finally:
        # 8. Всегда освобождаем модель после использования
        if model_gpu is not None:
            _release(model_gpu)
# --- Конец новой метрики ---

# --- НОВАЯ МЕТРИКА: BLIP Caption + BERTScore ---
@torch.inference_mode()
def get_blip_caption_bertscore(img: Image.Image, prompt: str) -> float:
    """
    Вычисляет BERTScore между описанием (caption), сгенерированным моделью BLIP,
    и исходным текстовым промптом.

    Args:
        img (PIL.Image.Image): Входное изображение.
        prompt (str): Исходный текстовый промпт.

    Returns:
        float: Среднее значение F1-меры BERTScore. Если `prompt` пуст,
               модель BLIP Caption не загружена, BERTScore недоступен
               или произошла ошибка, возвращает 0.0.
    """
    # Проверка, загружены ли необходимые компоненты
    if not BERT_SCORE_AVAILABLE:
        logger.debug("Библиотека `bert_score` недоступна. Возвращаю 0.0.")
        return 0.0

    if models.blip_cap_processor is None or models.blip_cap_model is None:
        logger.warning(
            "Модель или процессор BLIP Caption не загружены. "
            "Возвращаю 0.0 для BLIP Caption + BERTScore."
        )
        return 0.0

    prompt = prompt.strip()
    if not prompt:
        logger.debug("Исходный промпт для BLIP Caption + BERTScore пуст. Возвращаю 0.0.")
        return 0.0

    logger.debug(f"Вычисляю BLIP Caption + BERTScore для промпта: '{prompt[:50]}...'")

    try:
        # 1. Генерация описания (caption) с помощью BLIP
        inputs_for_caption = models.blip_cap_processor(images=img, return_tensors="pt")

        # 2. Перемещение модели на GPU (если доступен) и данных на то же устройство
        model_gpu = _to_gpu(models.blip_cap_model)
        inputs_moved = {k: v.to(model_gpu.device) for k, v in inputs_for_caption.items()}

        # 3. Прямой проход через модель генерации
        generated_ids = model_gpu.generate(**inputs_moved, max_new_tokens=MAX_SIG_TOK)

        # 4. Декодирование сгенерированного описания
        generated_caption = models.blip_cap_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        logger.debug(f"Сгенерированное описание BLIP: '{generated_caption[:100]}...'")

        if not generated_caption:
            logger.warning("Сгенерированное описание BLIP пусто. Возвращаю 0.0.")
            return 0.0

        # 5. Вычисление BERTScore между сгенерированным описанием и исходным промптом
        # bert_score.score возвращает (Precision, Recall, F1)
        P, R, F1 = bert_score_func(
            [generated_caption], # candidates (сгенерированный текст)
            [prompt],           # references (исходный промпт)
            lang='en',          # Язык (можно сделать параметром)
            verbose=False,      # Отключаем вывод прогресса
            device=model_gpu.device.type # Используем то же устройство, что и модель
        )

        # 6. Извлечение среднего F1-балла
        bert_score_value = F1.mean().item()

        logger.debug(
            f"BERTScore (P={P.mean().item():.4f}, R={R.mean().item():.4f}, F1={bert_score_value:.4f}) "
            f"между caption и prompt."
        )
        return bert_score_value

    except Exception as e:
        logger.error(f"Ошибка при вычислении BLIP Caption + BERTScore: {e}")
        return 0.0
    finally:
        # 7. Всегда освобождаем модель после использования
        if 'model_gpu' in locals():
            _release(model_gpu)

# --- Конец новой метрики ---

# --- Шаблон для новой метрики ---
# При добавлении новой метрики, создайте функцию по аналогии:
#
# @torch.inference_mode()
# def get_new_metric_score(img: Image.Image, ...) -> float:
#     """
#     Краткое описание новой метрики.
#
#     Args:
#         img (PIL.Image.Image): Входное изображение.
#         ... (другие аргументы): Аргументы, специфичные для метрики.
#
#     Returns:
#         float: Оценка по новой метрике.
#     """
#     # Реализация логики новой метрики
#     # Не забудьте использовать _to_gpu/_release, если используете
#     # модели, кроме Florence-2.
#     # ...
#     return score
#
# Затем интегрируйте её в `ranking.py`:
# 1. Добавьте импорт в `ranking.py`.
# 2. Вызовите её в цикле обработки изображений.
# 3. Добавьте её вес в аргументы CLI и `rank_folder`.
# 4. Включите её в нормализацию и вычисление итогового `total` балла.