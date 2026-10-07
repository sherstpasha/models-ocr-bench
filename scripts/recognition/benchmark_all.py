"""Run every word-recognition backend in its dependency-compatible environment."""

import argparse
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOBS = {
    "main": (".venv", "scripts.recognition.benchmark"),
    "paddleocr": (".venv-paddleocr", "scripts.recognition.benchmark_paddleocr"),
    "turkicocr": (".venv", "scripts.recognition.benchmark_turkicocr"),
    "kraken": (".venv-kraken", "scripts.recognition.benchmark_kraken"),
    "tesseract": (".venv", "scripts.recognition.benchmark_tesseract"),
    "crnn-ctc": (".venv", "scripts.recognition.benchmark_crnn_ctc"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["all", *JOBS],
        default="all",
        help="Run every backend or only one dependency-compatible group.",
    )
    parser.add_argument(
        "--include-high-power",
        action="store_true",
        help="Include models marked high-power. May stress GPU power and cooling.",
    )
    return parser.parse_args()


def interpreter(environment):
    path = PROJECT_ROOT / environment / "Scripts" / "python.exe"
    if not path.is_file():
        raise FileNotFoundError(f"Missing environment: {path}")
    return path


def main():
    args = parse_args()
    selected = JOBS if args.backend == "all" else {args.backend: JOBS[args.backend]}
    for index, (name, (environment, module)) in enumerate(selected.items(), start=1):
        python = interpreter(environment)
        print(
            f"\n[{index}/{len(selected)}] Running {name} with {python}",
            flush=True,
        )
        environment_variables = os.environ.copy()
        if not args.include_high_power:
            environment_variables["OCR_BENCH_SKIP_HIGH_POWER"] = "1"
        subprocess.run(
            [str(python), "-m", module],
            check=True,
            env=environment_variables,
        )

    python = interpreter(".venv")
    subprocess.run(
        [str(python), "-m", "scripts.recognition.summarize_results"],
        check=True,
    )
    print("\nAll word-recognition backends completed.")


if __name__ == "__main__":
    main()
