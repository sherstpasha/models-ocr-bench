# models-ocr-bench

GPU-бенчмарк моделей детекции текста.

## 1. Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements-gpu.txt
```

Веса моделей автоматически скачиваются при первом запуске в `models/`.

## 2. Датасеты

Датасеты скачиваются и распаковываются вручную в `C:\benchmark`:

```text
C:\benchmark\
├── YeniseiGovReports-TD\
├── school_notebooks_RU\
├── handwritten_essay\
├── ICDAR2015\
└── TotalText\
```

Ссылки:

- [YeniseiGovReports-TD](https://huggingface.co/datasets/anna4uonline/YeniseiGovReports-TD)
- [school_notebooks_RU](https://huggingface.co/datasets/ai-forever/school_notebooks_RU)
- [handwritten_essay](https://huggingface.co/datasets/sherstpasha/handwritten_essay)

Для `handwritten_essay` изображения с
[Mendeley Data](https://data.mendeley.com/datasets/vs44v8r3nf/1) и разметка с
Hugging Face автоматически скачиваются при первом запуске.

Подготовка нестандартных форматов:

```powershell
python -m scripts.prepare_school_notebooks
```

## 3. Запуск

```powershell
python -m scripts.benchmark
```

Уже рассчитанные пары модель/датасет автоматически пропускаются.

## 4. Результаты

- отдельные JSON: `benchmark_results/<model>/<dataset>_<model>.json`;
- общая CSV-таблица: `benchmark_results/summary.csv`;
- общая Markdown-таблица: `benchmark_results/summary.md`.

В сводной таблице строки — модели, столбцы — датасеты, значение —
`F1@0.5 / F1@0.5:0.95`.
