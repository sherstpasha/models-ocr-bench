"""Prepare datasets and run enabled line-level detection benchmarks."""

import argparse
import subprocess
from pathlib import Path

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.detection.prepare_school_notebooks import prepare_dataset as prepare_school_images
from scripts.line_detection.prepare_datasets import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ENVIRONMENTS = {
    "mask2former": ".venv",
    "rfdetr": ".venv-rfdetr",
    "doc_ufcn": ".venv-doc-ufcn",
    "paddleocr": ".venv-paddleocr",
    "surya": ".venv-surya",
    "kraken": ".venv-kraken",
    "pero": ".venv-pero",
    "rtmdet": ".venv-rtmdet",
    "ultralytics": ".venv",
}
BACKEND_ORDER = [
    "doc_ufcn", "rfdetr", "paddleocr", "surya", "mask2former",
    "kraken", "pero", "rtmdet", "ultralytics",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=sorted({config["backend"] for config in BENCHMARKS.values()}),
        help="Run only one dependency-compatible backend.",
    )
    return parser.parse_args()


def enabled_jobs(selected_backend=None):
    jobs = []
    seen = set()
    for config in BENCHMARKS.values():
        backend = config["backend"]
        if selected_backend is not None:
            if backend != selected_backend:
                continue
        elif not config.get("run") and backend != "paddleocr":
            continue
        module = f"scripts.line_detection.benchmark_{backend}"
        if module in seen:
            continue
        seen.add(module)
        environment = BACKEND_ENVIRONMENTS[backend]
        interpreter = PROJECT_ROOT / environment / "Scripts" / "python.exe"
        if not interpreter.is_file():
            if selected_backend is not None:
                raise FileNotFoundError(
                    f"Missing environment for {backend}: {interpreter}. "
                    "Create it according to README.md."
                )
            print(f"Skip {backend}: missing environment {interpreter}", flush=True)
            continue
        jobs.append((backend, module, interpreter))
    return sorted(jobs, key=lambda job: BACKEND_ORDER.index(job[0]))


def main():
    args = parse_args()
    for name, config in DATASETS.items():
        print(f"\nPreparing {name}...", flush=True)
        if config["prepare"] == "school_notebooks_lines":
            prepare_school_images(
                config["source_annotations"].parent,
                config["folder"].parent,
            )
        prepare_dataset(name, config)
        if not config["folder"].is_dir() or not config["annotations"].is_file():
            raise FileNotFoundError(f"Dataset is not ready: {name}")
    jobs = enabled_jobs(args.backend)
    for index, (backend, module, interpreter) in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] Running {backend} with {interpreter}",
            flush=True,
        )
        subprocess.run([str(interpreter), "-m", module], check=True)
    # Optional backend environments intentionally contain only their model's
    # dependencies. Build shared reports with the main environment where
    # OpenCV and all visualization dependencies are installed.
    main_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not main_python.is_file():
        raise FileNotFoundError(f"Missing main benchmark environment: {main_python}")
    subprocess.run(
        [str(main_python), "-m", "scripts.line_detection.summarize_results"],
        check=True,
    )
    subprocess.run(
        [str(main_python), "-m", "scripts.line_detection.visualize_predictions"],
        check=True,
    )


if __name__ == "__main__":
    main()
