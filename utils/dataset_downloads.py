import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset_files(name, config):
    repository = config.get("repository")
    files = config.get("download_files", [])
    download_dir = config.get("download_dir")
    if repository and files and download_dir is not None:
        missing = [
            filename
            for filename in files
            if not (download_dir / filename).is_file()
        ]
        if missing:
            print(f"Downloading {name} files from Hugging Face...", flush=True)
            download_dir.mkdir(parents=True, exist_ok=True)
            for filename in missing:
                hf_hub_download(
                    repo_id=repository,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=download_dir,
                )

    download_dataset_archive(name, config)
    download_huggingface_snapshot(name, config)
    download_huggingface_archive(name, config)
    download_kaggle_archive(name, config)
    extract_supported_archives(config)


def _safe_extract_path(root, member):
    root = Path(root).resolve()
    target = (root / member).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe archive member: {member}")


def extract_supported_archives(config):
    """Extract ZIP/TAR inputs idempotently; RAR remains an explicit manual step."""
    if config.get("prepare") == "page_xml_lines":
        # These split archives need task-specific routing into images/page_xmls.
        return
    destination = config.get("download_dir")
    if destination is None:
        return
    names = list(config.get("download_files", []))
    for key in ("archive_name", "kaggle_archive"):
        if config.get(key):
            names.append(config[key])
    for name in dict.fromkeys(names):
        archive = destination / name
        if not archive.is_file():
            continue
        marker = archive.with_name(f".{archive.name}.extracted")
        signature = f"{archive.stat().st_size}:{archive.stat().st_mtime_ns}"
        if marker.is_file() and marker.read_text(encoding="utf-8") == signature:
            continue
        lower = archive.name.lower()
        if lower.endswith(".zip"):
            print(f"Extracting {archive}...", flush=True)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    _safe_extract_path(destination, member.filename)
                bundle.extractall(destination)
            marker.write_text(signature, encoding="utf-8")
        elif lower.endswith((".tar", ".tar.gz", ".tgz")):
            print(f"Extracting {archive}...", flush=True)
            with tarfile.open(archive) as bundle:
                for member in bundle.getmembers():
                    _safe_extract_path(destination, member.name)
                bundle.extractall(destination)
            marker.write_text(signature, encoding="utf-8")


def download_huggingface_snapshot(name, config):
    repository = config.get("repository")
    destination = config.get("download_dir")
    if not repository or not config.get("huggingface_snapshot") or destination is None:
        return
    if config["folder"].is_dir() and config["source_labels"].is_file():
        return

    print(f"Downloading {name} from Hugging Face...", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        local_dir=destination,
    )
    if not config["folder"].is_dir() or not config["source_labels"].is_file():
        raise FileNotFoundError(
            f"Downloaded {repository} into {destination}, but expected images: "
            f"{config['folder']}; labels: {config['source_labels']}"
        )


def download_huggingface_archive(name, config):
    repository = config.get("repository")
    archive_name = config.get("huggingface_archive")
    destination = config.get("download_dir")
    if not repository or not archive_name or destination is None:
        return
    if config["folder"].is_dir() and config["source_labels"].is_file():
        return

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / archive_name
    if not archive.is_file():
        archive = Path(
            hf_hub_download(
                repo_id=repository,
                filename=archive_name,
                repo_type="dataset",
                local_dir=destination,
            )
        )
    # ZIP extraction is handled centrally by extract_supported_archives().


def download_kaggle_archive(name, config):
    dataset = config.get("kaggle_dataset")
    archive_name = config.get("kaggle_archive")
    destination = config.get("download_dir")
    if not dataset or not archive_name or destination is None:
        return
    if config["folder"].is_dir() and config["source_labels"].is_file():
        return

    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / archive_name
    if not archive.is_file():
        print(f"Downloading {name} from Kaggle...", flush=True)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "kaggle",
                    "datasets",
                    "download",
                    dataset,
                    "--path",
                    str(destination),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Kaggle download failed. Authenticate once with: kaggle auth login"
            ) from error

    if not archive.is_file():
        raise FileNotFoundError(f"Kaggle archive was not created: {archive}")
    if not config["folder"].is_dir() or not config["source_labels"].is_file():
        raise FileNotFoundError(
            f"Downloaded {archive}. Extract it manually into {destination}. "
            f"Expected images: {config['folder']}; "
            f"labels: {config['source_labels']}"
        )


def download_dataset_archive(name, config):
    url = config.get("archive_url")
    expected_hash = config.get("archive_sha256")
    destination = config.get("download_dir")
    if not url or not expected_hash or destination is None:
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
