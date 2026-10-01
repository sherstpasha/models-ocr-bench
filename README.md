# models-ocr-bench

Бенчмарки моделей детекции текста на GPU:

- `east_50_g1`;
- `yolo26s_obb_text_g1` и `yolo26x_obb_text_g1` из `manuscript-ocr`;
- `RoyRud1902/yolo11n-text`;
- `Daniil-Domino/yolo11x-dialectic`.
- `craft_easyocr` — детектор CRAFT из EasyOCR.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Для запуска EAST на NVIDIA GPU установите зависимости ONNX Runtime:

```powershell
python -m pip uninstall -y onnxruntime
python -m pip install -r requirements-gpu.txt
```

## Запуск

Пути к датасетам и параметры модели задаются в
`configs/benchmark_config.py`.

Модели `manuscript-ocr`:

```powershell
python -m scripts.benchmark_east_50_g1
```

Модели Ultralytics с Hugging Face:

```powershell
python -m scripts.benchmark_yolo
```

Весовые файлы с Hugging Face автоматически скачиваются в `models/` при первом
запуске и повторно используются в последующих запусках.

CRAFT из EasyOCR:

```powershell
python -m scripts.benchmark_craft_easyocr
```

Веса CRAFT автоматически скачиваются в `models/easyocr` при первом запуске.

Результаты каждой модели сохраняются в отдельном каталоге `benchmark_results/`.
