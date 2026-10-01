# models-ocr-bench

Бенчмарки моделей детекции текста. Сейчас в конфигурации активна модель
`east_50_g1`; скрипт `benchmark_yolo.py` сохранён для последующего подключения
YOLO-моделей.

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

```powershell
python scripts/benchmark_east_50_g1.py
```

Результаты сохраняются в `benchmark_results/east_50_g1`.
