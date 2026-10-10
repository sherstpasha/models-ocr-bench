"""Prepare line crops and run line-level recognition benchmarks."""

import argparse
import os
import subprocess
from pathlib import Path

os.environ["OCR_BENCH_RECOGNITION_LEVEL"] = "line"

from configs.recognition.benchmark_config import DATASETS  # noqa: E402
from scripts.recognition.prepare_handwritten_essay import prepare_dataset  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS = {
    "main": (".venv", "scripts.recognition.benchmark"),
    "paddleocr": (".venv-paddleocr", "scripts.recognition.benchmark_paddleocr"),
    "turkicocr": (".venv", "scripts.recognition.benchmark_turkicocr"),
    "kraken": (".venv-kraken", "scripts.recognition.benchmark_kraken"),
    "party": (".venv-party", "scripts.recognition.benchmark_party"),
    "tesseract": (".venv", "scripts.recognition.benchmark_tesseract"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["all", *JOBS],
        default="all",
        help="Run every backend or only one dependency-compatible group.",
    )
    return parser.parse_args()


def interpreter(environment):
    path = PROJECT_ROOT / environment / "Scripts" / "python.exe"
    if not path.is_file():
        raise FileNotFoundError(f"Missing environment: {path}")
    return path


def main():
    args = parse_args()
    for name, config in DATASETS.items():
        print(f"\nPreparing {name}...", flush=True)
        prepare_dataset(name, config)

    selected = JOBS if args.backend == "all" else {args.backend: JOBS[args.backend]}
    for index, (name, (environment, module)) in enumerate(selected.items(), start=1):
        python = interpreter(environment)
        print(f"\n[{index}/{len(selected)}] Running {name} with {python}", flush=True)
        subprocess.run([str(python), "-m", module], check=True)

    python = interpreter(".venv")
    subprocess.run(
        [str(python), "-m", "scripts.line_recognition.summarize_results"],
        check=True,
    )


if __name__ == "__main__":
    main()
