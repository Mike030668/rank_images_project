# Rank Images Project

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Инструмент для ранжирования изображений на основе текстовых промптов и внутреннего качества, используя современные модели искусственного интеллекта: **SigLIP-2**, **Florence-2**, **CLIP-IQA** и **DINOv2**.

## Содержание

- [Описание](#описание)
- [Возможности](#возможности)
- [Установка](#установка)
- [Использование](#использование)
  - [Быстрый старт с демонстрацией](#быстрый-старт-с-демонстрацией)
  - [Ранжирование ваших изображений](#ранжирование-ваших-изображений)
  - [Аргументы командной строки](#аргументы-командной-строки)
- [Структура проекта](#структура-проекта)
- [Лицензия](#лицензия)

## Описание

Этот проект предоставляет скрипт командной строки (`rank-images`), который позволяет упорядочить коллекцию изображений. Ранжирование происходит на основе комбинированного балла, рассчитанного из четырех метрик:

1.  **Схожесть изображения и текста (SigLIP-2):** Оценивает, насколько изображение соответствует позитивным текстовым запросам и отличается от негативных.
2.  **Поиск объектов по запросу (Florence-2):** Проверяет, насколько успешно Florence-2 может обнаружить объекты, описанные в позитивных промптах, на изображении.
3.  **Общее качество изображения (CLIP-IQA):** Дает оценку общего визуального качества изображения.
4.  **Внутренняя структура (DINOv2):** Косвенно оценивает "структурированность" или богатство деталей изображения.

Каждая метрика имеет свой вес (alpha, beta, gamma, delta), который можно настроить. Модели оптимизированы для работы как на GPU (с экономией VRAM), так и на CPU.

## Возможности

-   **Экономия VRAM:** Модели SigLIP-2 и DINOv2 временно загружаются на GPU только во время вычислений. Florence-2 использует автоматическое распределение (`device_map="auto"`).
-   **Поддержка GPU и CPU:** Автоматическое определение наличия CUDA.
-   **Гибкие промпты:** Поддержка JSON-файлов с позитивными/негативными промптами или простых текстовых строк.
-   **Разбиение длинных текстов:** Автоматическое разбиение длинных промптов для SigLIP-2.
-   **Расширяемость:** Код структурирован для легкого добавления новых метрик.

## Установка

Для установки проекта рекомендуется использовать `pip` или `uv pip`.

**Важно:** Проект требует **PyTorch >= 2.7.1**. Для его установки необходимо использовать **nightly** сборки с индекса PyTorch.

### Вариант 1: Используя `pip`

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/Mike030668/rank_images_project.git
cd rank_images_project

# Обновим инструменты на всякий случай
!pip install --upgrade pip setuptools wheel

# 2. (Рекомендуется) Создайте виртуальное окружение
python -m venv venv
# Активируйте его:
#   Linux/macOS: source venv/bin/activate
#   Windows:     venv\Scripts\activate

# 3. Установите проект в режиме редактирования (editable mode) с основными зависимостями
#    Используем --pre и дополнительный индекс для установки nightly PyTorch
pip install --pre \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu121 \
  -e .

# 4. (Опционально) Установите дополнительные зависимости
# Для Jupyter ноутбуков:
# pip install -e .[notebook]
```

### Вариант 2: Используя `uv pip` (если установлен `uv`)

`uv` значительно ускоряет установку зависимостей.

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/Mike030668/rank_images_project.git
cd rank_images_project

# 2. (Рекомендуется) Создайте виртуальное окружение с помощью uv
uv venv

# 3. Активируйте его:
#   Linux/macOS: source .venv/bin/activate
#   Windows:     .venv\Scripts\activate

# 4. Установите проект и зависимости, указав дополнительный индекс для PyTorch

uv pip install --prerelease=allow \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu121 \
  --index-strategy unsafe-best-match \
  -e .


# 5. (Опционально) Установите дополнительные зависимости
# uv pip install -e .[notebook]
```

## Использование

После установки вы можете использовать команду `rank-images` из любого места, если ваше виртуальное окружение активно.

### Быстрый старт с демонстрацией

Проект включает демонстрационные изображения и промпты. Чтобы быстро проверить работу:

```bash
rank-images --demo
```

Это выполнит ранжирование изображений из папки `data/demo_images/` с использованием `data/demo_images/prompts.json`. Результат будет сохранен в `data/demo_images/ranking.csv`.

### Ранжирование ваших изображений

```bash
rank-images /путь/к/вашей/папке/с/изображениями --prompts /путь/к/prompts.json
```

**Пример:**

1.  Положите ваши изображения (`.jpg`, `.png`) в папку, например, `my_images/`.
2.  Создайте файл `my_prompts.json`:

    ```json
    {
      "prompt": "clear photo of a modern skyscraper, blue sky, high resolution",
      "prompt2": "sharp focus, detailed architecture",
      "negative": "blurry, low resolution, watermark, cartoon",
      "negative2": "night time, dark"
    }
    ```
3.  Запустите ранжирование:

    ```bash
    rank-images my_images --prompts my_prompts.json
    ```

4.  Результат будет записан в `my_images/ranking.csv`. Изображения будут отсортированы по убыванию итогового балла (`total`).

### Аргументы командной строки

```bash
rank-images --help
```

```
usage: rank-images [-h] [--prompts PROMPTS] [--alpha ALPHA] [--beta BETA]
                   [--gamma GAMMA] [--delta DELTA] [--chunk CHUNK] [--demo]
                   img_dir

Ранжирует изображения в заданной директории на основе текстовых промптов и
внутреннего качества, используя SigLIP-2, Florence-2, CLIP-IQA и DINOv2.

positional arguments:
  img_dir               Путь к директории с изображениями для ранжирования.

options:
  -h, --help            show this help message and exit
  --prompts PROMPTS     Источник текстовых промптов. Может быть:
                        - Путь к .json файлу с ключами 'prompt', 'prompt2',
                          'negative', 'negative2'.
                        - Произвольная текстовая строка (будет использована
                          как 'prompt').
                        Если не указан, ранжирование будет основано только на
                        именах файлов.
  --alpha ALPHA         Вес метрики SigLIP (схожесть изображения и текста).
                        По умолчанию 0.6.
  --beta BETA           Вес метрики Florence-2 (поиск объектов по запросу).
                        По умолчанию 0.4.
  --gamma GAMMA         Вес метрики CLIP-IQA (общее качество изображения).
                        По умолчанию 0.2.
  --delta DELTA         Вес метрики DINOv2 (внутренние признаки изображения).
                        По умолчанию 0.1.
  --chunk CHUNK         Максимальное количество токенов в одном фрагменте
                        текста для SigLIP.
                        Используется для разбиения длинных текстов. По
                        умолчанию используется значение из config.py.
  --demo                Запустить демонстрацию на встроенных примерах из
                        data/demo_images/.
```

## Структура проекта

Структура проекта следует рекомендациям Cookiecutter Data Science для организации кода и данных.

```
rank_images_project/
├── README.md                 # Этот файл
├── pyproject.toml            # Конфигурация проекта и зависимостей
├── .gitignore                # Файлы и папки, игнорируемые Git
├── data/
│   ├── raw/                  # (Опционально) Ваши собственные изображения для обработки
│   └── demo_images/          # Демонстрационные изображения и промпты
│       ├── image1.jpg        # Пример изображения 1
│       ├── image2.png        # Пример изображения 2
│       └── prompts.json      # Промпты для демонстрационных изображений
├── notebooks/                # Jupyter ноутбуки для исследований
├── src/
│   └── rank_images/          # Основной пакет Python
│       ├── __init__.py       # Делает каталог пакетом
│       ├── config.py         # Глобальные константы
│       ├── device_utils.py   # Функции управления устройствами (_to_gpu, _release)
│       ├── models.py         # Загрузка и хранение моделей
│       ├── data_processing.py# Вспомогательные функции обработки данных
│       ├── metrics.py        # Функции вычисления метрик
│       ├── ranking.py        # Основная логика ранжирования
│       └── cli.py            # Интерфейс командной строки
└── outputs/                  # (Игнорируется) Для будущих результатов/промежуточных данных
```

## Лицензия

Этот проект лицензирован по лицензии MIT - см. файл [LICENSE](LICENSE) для подробностей.
