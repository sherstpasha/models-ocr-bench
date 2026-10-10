"""Cross-platform dataset, benchmark, and summary orchestration."""

import argparse
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("detection", "line-detection", "recognition", "line-recognition")


def environment_python(name):
    relative = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    path = PROJECT_ROOT / name / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing environment interpreter: {path}")
    return path


def run_process(label, environment, module, log, extra_env=None):
    command = [str(environment_python(environment)), "-m", module]
    values = os.environ.copy()
    values.update(extra_env or {})
    print(f"\n## {label}\n{' '.join(command)}", flush=True)
    log.write(f"\n## {label}\n{' '.join(command)}\n")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=values,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
        log.flush()
    return process.wait()


def model_units(task):
    if task == "detection":
        from configs.detection.benchmark_config import BENCHMARKS
        environment = {"paddleocr": ".venv-paddleocr"}
        return [
            (name, environment.get(config["backend"], ".venv"),
             f"scripts.detection.{config['script'].stem}")
            for name, config in BENCHMARKS.items()
        ]
    if task == "line-detection":
        from configs.line_detection.benchmark_config import BENCHMARKS
        environments = {
            "mask2former": ".venv", "rfdetr": ".venv-rfdetr",
            "doc_ufcn": ".venv-doc-ufcn", "paddleocr": ".venv-paddleocr",
            "surya": ".venv-surya", "kraken": ".venv-kraken",
            "orli": ".venv-orli",
            "pero": ".venv-pero", "rtmdet": ".venv-rtmdet",
            "ultralytics": ".venv",
        }
        return [
            (name, environments[config["backend"]],
             f"scripts.line_detection.benchmark_{config['backend']}")
            for name, config in BENCHMARKS.items()
        ]

    from configs.recognition.benchmark_config import BENCHMARKS
    modules = {
        "trba": (".venv", "scripts.recognition.benchmark_trba"),
        "easyocr": (".venv", "scripts.recognition.benchmark_easyocr"),
        "trocr": (".venv", "scripts.recognition.benchmark_trocr"),
        "parseq": (".venv", "scripts.recognition.benchmark_parseq"),
        "paddleocr": (".venv-paddleocr", "scripts.recognition.benchmark_paddleocr"),
        "turkicocr_onnx": (".venv", "scripts.recognition.benchmark_turkicocr"),
        "kraken": (".venv-kraken", "scripts.recognition.benchmark_kraken"),
        "tesseract": (".venv", "scripts.recognition.benchmark_tesseract"),
        "crnn_ctc": (".venv", "scripts.recognition.benchmark_crnn_ctc"),
        "party": (".venv-party", "scripts.recognition.benchmark_party"),
    }
    return [(name, *modules[config["backend"]]) for name, config in BENCHMARKS.items()]


def prepare_task(task):
    if task == "detection":
        from scripts.detection.benchmark import prepare_datasets, validate_datasets
        prepare_datasets()
        validate_datasets()
    elif task == "line-detection":
        from configs.line_detection.benchmark_config import DATASETS
        from scripts.detection.prepare_school_notebooks import prepare_dataset as prepare_school
        from scripts.line_detection.prepare_datasets import prepare_dataset
        for name, config in DATASETS.items():
            if config["prepare"] == "school_notebooks_lines":
                prepare_school(config["source_annotations"].parent, config["folder"].parent)
            prepare_dataset(name, config)


def run_task(task, log, errors):
    print(f"\n# TASK: {task}", flush=True)
    try:
        prepare_task(task)
    except Exception:
        detail = traceback.format_exc()
        print(detail)
        log.write(detail)
        errors.append((f"{task}: dataset preparation", "failed"))
        return

    level = "line" if task == "line-recognition" else "word"
    for model, environment, module in model_units(task):
        code = run_process(
            f"{task} / {model}", environment, module, log,
            {"OCR_BENCH_ONLY_MODEL": model, "OCR_BENCH_RECOGNITION_LEVEL": level},
        )
        if code:
            errors.append((f"{task} / {model}", f"exit code {code}"))


SUMMARY_MODULES = {
    "detection": "scripts.detection.summarize_results",
    "line-detection": "scripts.line_detection.summarize_results",
    "recognition": "scripts.recognition.summarize_results",
    "line-recognition": "scripts.line_recognition.summarize_results",
}


def update_summaries(tasks, log, errors):
    for task in tasks:
        code = run_process(
            f"summary / {task}", ".venv", SUMMARY_MODULES[task], log,
            {"OCR_BENCH_RECOGNITION_LEVEL": "line" if task == "line-recognition" else "word"},
        )
        if code:
            errors.append((f"summary / {task}", f"exit code {code}"))
    code = run_process("README", ".venv", "scripts.build_readme", log)
    if code:
        errors.append(("README", f"exit code {code}"))


def finish(log_path, errors):
    print(f"\nLog: {log_path}")
    if errors:
        print("\nCompleted with errors:")
        for label, detail in errors:
            print(f"- {label}: {detail}")
        print("See the log above for complete tracebacks.")
    else:
        print("\nCompleted without errors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download-datasets")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("task", choices=("all", *TASKS))
    summary_parser = subparsers.add_parser("summary")
    summary_parser.add_argument("task", choices=("all", *TASKS))
    args = parser.parse_args()

    if args.command == "download-datasets":
        from scripts.download_datasets import main as download
        download()
        return

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{args.command}_{args.task}_{stamp}.log"
    tasks = TASKS if args.task == "all" else (args.task,)
    errors = []
    with log_path.open("w", encoding="utf-8") as log:
        if args.command == "run":
            for task in tasks:
                run_task(task, log, errors)
            update_summaries(tasks, log, errors)
        else:
            update_summaries(tasks, log, errors)
        if errors:
            log.write("\nERROR SUMMARY\n")
            for label, detail in errors:
                log.write(f"- {label}: {detail}\n")
    finish(log_path, errors)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
