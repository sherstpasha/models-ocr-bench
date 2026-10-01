import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.detection.summarize_results import write_summary


def enabled_modules():
    modules = []
    for config in BENCHMARKS.values():
        if not config.get("run", False):
            continue
        module = f"scripts.detection.{config['script'].stem}"
        if module not in modules:
            modules.append(module)
    return modules


def download_dataset_files():
    for name, config in DATASETS.items():
        repository = config.get("repository")
        files = config.get("download_files", [])
        download_dir = config.get("download_dir")
        if not repository or not files or download_dir is None:
            continue

        missing = [
            filename
            for filename in files
            if not (download_dir / filename).is_file()
        ]
        if missing:
            print(f"Downloading {name} annotations from Hugging Face...", flush=True)
            download_dir.mkdir(parents=True, exist_ok=True)
            for filename in missing:
                hf_hub_download(
                    repo_id=repository,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=download_dir,
                )

        download_dataset_archive(name, config)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            target = (destination / item.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {item.filename}")
        zipped.extractall(destination)


def download_dataset_archive(name, config):
    url = config.get("archive_url")
    expected_hash = config.get("archive_sha256")
    destination = config.get("download_dir")
    image_folder = config.get("folder")
    if not url or not expected_hash or destination is None or image_folder.exists():
        return

    archive = destination / config["archive_name"]
    partial = Path(f"{archive}.part")
    destination.mkdir(parents=True, exist_ok=True)

    if partial.is_file():
        if sha256(partial) == expected_hash:
            os.replace(partial, archive)
        else:
            partial.unlink()

    if not archive.is_file() or sha256(archive) != expected_hash:
        print(f"Downloading {name} images from Mendeley Data...", flush=True)
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256(partial) != expected_hash:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {name} archive")
        os.replace(partial, archive)

    print(f"Extracting {name} images...", flush=True)
    safe_extract(archive, destination)
    if not image_folder.exists():
        raise FileNotFoundError(f"Archive did not contain {image_folder}")


def main():
    download_dataset_files()

    modules = enabled_modules()
    if not modules:
        print("No benchmarks enabled in configs/detection/benchmark_config.py")
        return

    for index, module in enumerate(modules, start=1):
        print(f"\n[{index}/{len(modules)}] Running {module}", flush=True)
        subprocess.run([sys.executable, "-m", module], check=True)

    write_summary()
    print("\nAll enabled benchmarks completed.")


if __name__ == "__main__":
    main()
