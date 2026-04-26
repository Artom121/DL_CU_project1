# SIMILIS Baseline: формализованное описание артефакта по изображению

Этот репозиторий содержит воспроизводимый baseline для задачи SIMILIS: предсказать визуальные атрибуты археологического артефакта по изображению и собрать поле `auto_description` по фиксированному шаблону

## Структура проекта

- `src/similis_baseline` - основной код (нормализация меток, data pipeline, модель, обучение, инференс, шаблон описания).
- `notebooks` - пошаговые baseline-ноутбуки (`01`..`05`) для setup, проверки данных, EDA, выбора целей и проверки split/preprocessing/dataloader.
- `configs` - конфиги экспериментов (`baseline` и ablation).
- `tools` - утилиты (оценка, sanity-forward, запуск ноутбуков).
- `train.py` - входная точка обучения модели.
- `predict.py` - входная точка инференса по директории изображений.
- `data` - сырые/промежуточные/обработанные данные.
- `splits` - сохраненные group-aware split-файлы.
- `artifacts` - чекпоинты, предсказания, отчеты и фигуры.

## Окружение

Создание Conda-окружения из `environment.yml`:

```bash
conda env create -f environment.yml
conda activate similis-baseline
```

## Данные

Cтруктура датасета:

- `data/raw/cu_data/...`
- CSV с метаданными внутри распакованного архива, обычно `selected_by_name_iimk_subset_public.csv`

Основной манифест для обучения:

- `data/interim/manifest_raw.csv`

## Обучение

Базовый запуск обучения:

```bash
python train.py \
  --config configs/baseline_resnet18_224.json \
  --manifest data/interim/manifest_raw.csv \
  --output-dir artifacts/checkpoints/baseline_resnet18_224
```

Быстрый sanity-запуск:

```bash
python train.py \
  --config configs/baseline_resnet18_224.json \
  --manifest data/interim/manifest_raw.csv \
  --output-dir artifacts/checkpoints/tiny_overfit_resnet18_224 \
  --tiny-overfit
```

## Инференс

Генерация предсказаний для директории изображений:

```bash
python predict.py \
  --image-dir data/raw/cu_data/cu_data/images \
  --checkpoint artifacts/checkpoints/tiny_overfit_resnet18_224/best.pt \
  --output-csv artifacts/preds/inference_example.csv \
  --device cpu
```

Выходной CSV содержит:

- `image_file` - относительный путь к изображению
- `auto_description` - сгенерированное нормализованное описание

При необходимости в CSV можно добавлять вспомогательные колонки для анализа (предсказанные поля и confidence).

## Оценка и артефакты

- Логи обучения и summary запусков: `artifacts/checkpoints/*/{train_log.csv,summary.json}`
- Предсказания: `artifacts/preds/*.csv`
- Отчеты: `artifacts/reports/*`
- Фигуры: `artifacts/figures/*`

## Базовый метод

1. Построение нормализованных целевых полей (по умолчанию: `object_type`, `integrity`, `material_group`, `part_zone`).
2. Обучение multi-head классификатора по изображению.
3. Применение порогов уверенности по каждому полю.
4. Сборка `auto_description` по детерминированным шаблонным правилам.