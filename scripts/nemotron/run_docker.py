"""Build and run the Linux/CUDA Nemotron OCR v2 benchmark container."""

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "models-ocr-bench-nemotron:1.0.0"
MODEL_ROOT = PROJECT_ROOT / "models" / "nemotron-ocr-v2"
REQUIRED_FILES = (
    MODEL_ROOT / "nemotron-ocr" / "pyproject.toml",
    MODEL_ROOT / "v2_multilingual" / "detector.pth",
    MODEL_ROOT / "v2_multilingual" / "recognizer.pth",
    MODEL_ROOT / "v2_multilingual" / "relational.pth",
    MODEL_ROOT / "v2_multilingual" / "charset.txt",
    MODEL_ROOT / "v2_multilingual" / "model_config.json",
)


def ensure_model_files():
    if all(path.is_file() for path in REQUIRED_FILES):
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="nvidia/nemotron-ocr-v2",
        local_dir=MODEL_ROOT,
        allow_patterns=["nemotron-ocr/**", "v2_multilingual/**"],
    )


def main():
    ensure_model_files()
    if subprocess.run(["docker", "image", "inspect", IMAGE], check=False,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run(
            ["docker", "build", "-f", "scripts/nemotron/Dockerfile", "-t", IMAGE, "."],
            cwd=PROJECT_ROOT,
            check=True,
        )
    level = os.environ.get("OCR_BENCH_RECOGNITION_LEVEL")
    task = "recognition" if level == "line" else "detection"
    command = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        "-e", "OCR_BENCH_DATA_ROOT=/data",
        "-e", f"OCR_BENCH_NEMOTRON_TASK={task}",
        "-v", f"{PROJECT_ROOT}:/workspace/bench",
        "-v", r"C:\benchmark:/data",
        IMAGE,
        "python", "-u", "-m", "scripts.nemotron.benchmark",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
