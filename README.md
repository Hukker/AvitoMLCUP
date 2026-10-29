# Baseline
## Запуск

Установите [uv](https://docs.astral.sh/uv/) и распакуйте датасет.
Создайте окружение один раз и активируйте его:

```bash
cd baselines
uv sync --locked --no-config --default-index https://pypi.org/simple
source .venv/bin/activate
```

Все зависимости публичные: используются только PyPI и файлы с
`files.pythonhosted.org`. Доступ к внутреннему PyPI Авито не нужен.
`--no-config` отключает чтение локальных конфигурационных файлов uv;
публичный индекс в команде указан явно. `--locked` сохраняет версии из `uv.lock`.

Все команды выполняются из **`baselines/`**. `--data` указывает на `competition/`.

```bash
python catboost_baseline.py train \
  --data /path/to/competition --output runs/catboost-01
python catboost_baseline.py predict \
  --data /path/to/competition --model runs/catboost-01/model.cbm \
  --output runs/catboost-01/submission.parquet

python neural_baseline.py train \
  --data /path/to/competition --output runs/neural-01
python neural_baseline.py predict \
  --data /path/to/competition --model runs/neural-01/model.pt \
  --output runs/neural-01/submission.parquet
```

- GPU включается перед подкомандой: `neural_baseline.py --device cuda train ...`.

## Предсказания на тестовой выборке

Скрипты — `catboost_baseline.py` и `neural_baseline.py`, подкоманда **`predict`**.
Она читает весь `competition/test/`; в `--data` нужно передать корень `competition/`,
а не папку `test/`. 

```bash
python catboost_baseline.py predict \
  --data /path/to/competition --model runs/catboost-01/model.cbm \
  --output runs/catboost-01/test-submission.parquet

python neural_baseline.py --device cuda predict \
  --data /path/to/competition --model runs/neural-01/model.pt \
  --output runs/neural-01/test-submission.parquet
```

Для NN на CPU замените `--device cuda` на `--device cpu`.
В `--model` укажите `model.pt`, сохранённый обучением.

Результат — Parquet с `feed_id, item_id, score` для всех кандидатов test
в исходном порядке. Больший score означает более высокую позицию в ленте.
Выберите новый путь `--output`: существующий submission не перезаписывается.

## Метрики public/private

`calculate_metrics.py` считает NDCG, NDCG@10 и NDCG@30 на всём test
и отдельно на public/private. Для расчёта нужен файл правильных ответов
`test_answers.parquet` с колонками `feed_id, item_id, target, leaderboard_split`.
Этот файл не используется при обучении и выборе checkpoint.

```bash
python calculate_metrics.py \
  --predictions runs/neural-01/submission.parquet \
  --answers /path/to/test_answers.parquet > runs/neural-01/metrics.json
```

Для CatBoost передайте его submission через `--predictions`.
Скрипт соединяет строки по `(feed_id, item_id)` и проверяет уникальность ключей,
полноту предсказаний и конечность scores. Порядок строк не важен,
`sample_submission` не требуется. Каждая лента целиком относится к public или private.

В JSON сохраняются метрики для `all`, `public`, `private`, число строк,
групп и позитивов. Метрики усредняются по лентам с одинаковым весом;
gain — бинарный target, discount — `1 / log2(position + 1)`.
При одинаковых scores позитивы получают худшие позиции;
для ленты без позитивов NDCG равен 1, как в CatBoost.
Результат дополнительно сверяется с независимым расчётом через NumPy.
GPU не требуется; зависимости уже входят в окружение baseline-ов.

## Данные и результат

По умолчанию читаются **полный train и весь val**. Объём задаётся
в `DATA_PARAMS` (CatBoost) и `TRAIN_CONFIG` (NN)
в начале соответствующего `baseline.py`; `0` снимает лимит.

Оба baseline используют готовые `f1..f5` — нормализованные интерполированные бины.
Обратного преобразования в тройки нет. Векторы int8 переводятся в float32 до
вычислений. В NN пользовательские векторы нормализуются один раз при загрузке. Item-эмбеддинги
читаются из `data.memmap` и нормализуются при обращении: хранение всей таблицы
в float32 потребовало бы 38,5 GiB только для векторов. Индексы находятся в Parquet.
Индекс `-1` означает отсутствующий вектор. `pos`, ID, timestamp и история не
используются как признаки; test не участвует в обучении и подборе параметров.

## CatBoost

`CatBoostRanker` с конфигом обычного ranker из `src/train_ranker/defaults.py`:
YetiRank, GPU, до 2000 деревьев, learning rate 0,4, L2=5, seed 666.
Каждый `feed_id` — отдельная группа. Модель выбирается по NDCG@30;
`FIT_PARAMS` задаёт early stopping после 50 итераций без улучшения, как в rec-ranker.
Параметры скопированы в `MODEL_PARAMS` и `FIT_PARAMS`, внутренних импортов нет.

Для обучения нужна CUDA GPU; карта выбирается через `CUDA_VISIBLE_DEVICES`.
Для CPU замените `MODEL_PARAMS["task_type"]` на `"CPU"` и удалите `devices`
и `gpu_ram_part`. Predict работает на CPU. Сохраняются `model.cbm` и `run.json`
с NDCG, NDCG@10 и NDCG@30. Score — сырое значение ранкера, не вероятность.
Прежние метрики Logloss-классификатора не характеризуют этот YetiRank baseline.

Девять признаков: `f1..f5`, категориальный `category_id`, dot и cosine эмбеддингов,
флаг отсутствующего item-эмбеддинга. Для missing и векторов нулевой длины cosine — 0.
Пути Parquet сортируются по имени; `map_batches` возвращает по одному набору
признаков на входную ленту в том же порядке. `unnest` не меняет строки,
`explode` разворачивает все списки совместно в исходном порядке кандидатов.

## Transformer

Стандартный `torch.nn.TransformerEncoder` без позиционных embeddings: токен
пользователя и все кандидаты ленты взаимодействуют через self-attention.
Это самостоятельная реализация на стандартных слоях, без внутренних блоков,
энкодеров, конфигов и весов. В варианте по умолчанию около 85 тысяч параметров.

| Часть | Реализация по умолчанию |
|---|---|
| Пользователь | L2-нормализация вектора из 128 чисел, Linear → 64 |
| Кандидат | `f1..f5`, нормализованный item-вектор, missing-флаг и embedding категории из 8 чисел; Linear → 64 |
| Контекст | 2 encoder-слоя, 4 головы, FFN 128, dropout 0,1 |
| Score | Linear 64 → 1; sigmoid при predict |

Словарь категорий строится только по train; неизвестная категория получает нулевой
embedding. Нулевые векторы остаются нулевыми после нормализации. Padding исключён
из attention-ключей, loss и submission. Перестановка кандидатов переставляет
scores с точностью до округления.

По умолчанию: 5 эпох, batch 32 целые ленты, AdamW, learning rate 0,001,
weight decay 0,0001, seed 42. Данные представлены стандартным PyTorch `Dataset`: один элемент — целая лента.
`DataLoader` собирает batch, `collate_feeds` добавляет padding. Перед эпохой
перемешиваются ленты. Loss — BCE
по настоящим кандидатам, без class weights и negative sampling. Лучшая эпоха
по val Logloss сохраняется в `model.pt` вместе со словарём и размерами модели;
`run.json` содержит историю эпох и число параметров. Загрузка — с `weights_only=True`.

Размеры модели, регуляризация, batch size и число эпох задаются в `TRAIN_CONFIG`. Attention имеет четыре головы,
FFN — двойную ширину. Predict восстанавливает размеры из checkpoint.
