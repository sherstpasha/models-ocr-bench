import subprocess
import sys
from pathlib import Path

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.detection.prepare_school_notebooks import prepare_dataset
from scripts.detection.summarize_results import write_summary
from scripts.detection.visualize_predictions import main as visualize_predictions
from utils.dataset_downloads import download_dataset_files as download_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def enabled_jobs():
    jobs = []
    seen = set()
    for config in BENCHMARKS.values():
        # PaddleOCR is intentionally disabled in the main environment, but the
        # all-in-one runner launches it from its isolated environment.
        if not config.get("run", False) and config.get("backend") != "paddleocr":
            continue
        module = f"scripts.detection.{config['script'].stem}"
        if module in seen:
            continue
        seen.add(module)
        interpreter = Path(sys.executable)
        if config.get("backend") == "paddleocr":
            interpreter = PROJECT_ROOT / ".venv-paddleocr" / "Scripts" / "python.exe"
            if not interpreter.is_file():
                raise FileNotFoundError(
                    f"Missing PaddleOCR environment: {interpreter}. "
                    "Create it according to README.md."
                )
        jobs.append((module, interpreter))
    return jobs


def download_dataset_files():
    for name, config in DATASETS.items():
        download_dataset(name, config)


def prepare_datasets():
    for name, config in DATASETS.items():
        if config.get("prepare") == "old_orthography":
            from scripts.prepare_old_orthography import prepare

            print(f"\nPreparing {name}...", flush=True)
            prepare(config)
        elif config.get("prepare") == "school_notebooks":
            print(f"\nPreparing {name}...", flush=True)
            prepare_dataset(config["download_dir"], config["folder"].parent)


def validate_datasets():
    missing = []
    for name, config in DATASETS.items():
        image_dir = config["folder"]
        annotations = config["annotations"]
        has_images = image_dir.is_dir() and any(
            path.is_file() for path in image_dir.rglob("*")
        )
        if not has_images:
            missing.append(f"{name} images: {image_dir}")
        if not annotations.is_file():
            missing.append(f"{name} annotations: {annotations}")
    if missing:
        raise FileNotFoundError(
            "Datasets are not ready. Extract downloaded archives manually or "
            "prepare manual datasets:\n- " + "\n- ".join(missing)
        )


def main():
    download_dataset_files()
    prepare_datasets()
    validate_datasets()

    jobs = enabled_jobs()
    if not jobs:
        print("No benchmarks enabled in configs/detection/benchmark_config.py")
        return

    for index, (module, interpreter) in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] Running {module} with {interpreter}",
            flush=True,
        )
        subprocess.run([str(interpreter), "-m", module], check=True)

    write_summary()
    visualize_predictions()
    print("\nAll enabled benchmarks completed.")


if __name__ == "__main__":
    main()
