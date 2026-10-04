# Воспроизведение бенчмарка

Команды выполняются из корня репозитория. Требуются Git, NVIDIA GPU с
подходящим драйвером, Python 3.10 и свободное место для восьми окружений и
весов моделей.

## 1. Установка окружений и загрузка моделей

Скрипт создаёт `.venv`, `.venv-paddleocr`, `.venv-rfdetr`,
`.venv-doc-ufcn`, `.venv-surya`, `.venv-kraken`, `.venv-pero` и
`.venv-rtmdet`, устанавливает зафиксированные зависимости и скачивает все
веса. Датасеты и инференс на этом этапе не запускаются.

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_all_windows.ps1
```

Скрипт использует `py -3.10`. Если Tesseract 5 отсутствует и доступен
`winget`, он также будет установлен автоматически.

### Linux

```bash
bash scripts/setup_all_linux.sh
```

По умолчанию используется `python3.10`. Другой совместимый интерпретатор
можно передать переменной:

```bash
PYTHON_BIN=/usr/bin/python3.10 bash scripts/setup_all_linux.sh
```

Если Tesseract отсутствует, скрипт попытается установить его через `apt-get`.

### Проверка

После успешного завершения должны существовать все восемь каталогов `.venv*`,
каталог `models/` и кэши моделей `manuscript-ocr`. Команду можно безопасно
повторять: существующие файлы не скачиваются заново.

## 2. Загрузка и распаковка датасетов

Корневой каталог задаётся переменной `OCR_BENCH_DATA_ROOT`. По умолчанию это
`C:\benchmark` на Windows и `./benchmark_data` на Linux.

Windows:

```powershell
$env:OCR_BENCH_DATA_ROOT = "D:\ocr-benchmark-data"
.\.venv\Scripts\python.exe -m scripts.pipeline download-datasets
```

Linux:

```bash
export OCR_BENCH_DATA_ROOT=/data/ocr-benchmark
.venv/bin/python -m scripts.pipeline download-datasets
```

Команда охватывает датасеты всех четырёх задач. ZIP, TAR, TAR.GZ и TGZ
распаковываются автоматически и безопасно. Повторная распаковка неизменившихся
архивов пропускается.

### Ручные действия

- `YeniseiGovReports-HWR.rar` и `YeniseiGovReports-PRT.rar` скачиваются
  автоматически, но RAR нужно распаковать вручную. Команда напечатает точные
  ожидаемые пути `val/img` и `val/labels.csv`.
- [ICDAR2015](https://rrc.cvc.uab.es/?ch=4) нужно скачать вручную и положить
  `test_images/` и `test.json` в `$OCR_BENCH_DATA_ROOT/ICDAR2015/`.
- [TotalText](https://github.com/cs-chan/Total-Text-Dataset) нужно скачать
  вручную и положить `test_images/` и `test.json` в
  `$OCR_BENCH_DATA_ROOT/TotalText/`.
- Kaggle может запросить авторизацию для Cyrillic Handwriting. В таком случае
  выполните `kaggle auth login` и повторите общую команду.

В конце загрузчик всегда печатает единый список того, что ещё требует ручного
действия. Рабочая разметка и кропы создаются автоматически при запуске задач.

## 3. Запуск бенчмарков

Одна команда для всех четырёх задач:

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline run all
```

Linux использует тот же CLI:

```bash
.venv/bin/python -m scripts.pipeline run all
```

Отдельные задачи:

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline run detection
.\.venv\Scripts\python.exe -m scripts.pipeline run line-detection
.\.venv\Scripts\python.exe -m scripts.pipeline run recognition
.\.venv\Scripts\python.exe -m scripts.pipeline run line-recognition
```

Каждая модель запускается отдельным процессом в своём окружении. Ошибка одной
модели не останавливает остальные. Полный stdout, stderr и traceback сохраняются
в `logs/run_<задача>_<дата-время>.log`; в конце команды выводится краткий список
моделей с ошибками. Уже готовые результаты пропускаются существующими проверками.

## 4. Обновление сводных таблиц без инференса

Одна выбранная таблица:

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline summary recognition
```

Вместо `recognition` можно указать `detection`, `line-detection` или
`line-recognition`.

Все четыре таблицы:

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline summary all
```

Обе команды после обновления summary автоматически пересобирают таблицы в
`README.md`. На Linux замените `.\.venv\Scripts\python.exe` на
`.venv/bin/python`.
