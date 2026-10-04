"""Benchmark PERO OCR's published general layout-analysis model."""

import configparser
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser
from tqdm import tqdm

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run

MODEL_NAME = "pero_layout_general"
CONFIG = BENCHMARKS[MODEL_NAME]


def _download_and_extract(config):
    target = Path(config["model_dir"])
    configs = list(target.rglob("*.ini")) if target.is_dir() else []
    if configs:
        return configs
    target.mkdir(parents=True, exist_ok=True)
    archive = target / config["archive_name"]
    if not archive.is_file():
        print(f"Downloading PERO model archive to {archive}...", flush=True)
        with urllib.request.urlopen(config["archive_url"]) as response, archive.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc=archive.name) as progress:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    progress.update(len(chunk))
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise RuntimeError(f"Unsafe PERO archive member: {member.filename}")
        bundle.extractall(target)
    return list(target.rglob("*.ini"))


def create_model(config):
    candidates = _download_and_extract(config)
    selected = None
    parser_config = None
    for path in candidates:
        value = configparser.ConfigParser()
        value.read(path, encoding="utf-8")
        if value.has_section("PAGE_PARSER") and value.has_section("LAYOUT_PARSER_1"):
            selected, parser_config = path, value
            break
    if selected is None:
        raise FileNotFoundError(f"No PERO pipeline .ini found under {config['model_dir']}")
    parser_config.set("PAGE_PARSER", "RUN_LAYOUT_PARSER", "true")
    for option in ("RUN_LINE_CROPPER", "RUN_OCR", "RUN_DECODER"):
        parser_config.set("PAGE_PARSER", option, "false")
    parser = PageParser(parser_config, device=torch.device(config["device"]), config_path=str(selected.parent))
    return parser


def detect(parser, image_path, _config):
    raw = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    page = PageLayout(id=Path(image_path).stem, page_size=image.shape[:2])
    page = parser.process_page(image, page)
    polygons = []
    for line in page.lines_iterator():
        if line.polygon is not None and len(line.polygon) >= 3:
            polygons.append([np.asarray(line.polygon, dtype=np.float32).reshape(-1, 2).tolist()])
    return polygons


if __name__ == "__main__":
    run(MODEL_NAME, CONFIG, create_model, detect, "PERO general layout")
