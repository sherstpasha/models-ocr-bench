"""Download every model artifact without running benchmark inference."""

import argparse
import gc
import os
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("main", "paddleocr", "rfdetr", "ultralytics-line", "doc-ufcn", "surya", "kraken", "pero", "rtmdet")


def announce(name):
    print(f"\nDownloading/preparing: {name}", flush=True)


def download_main():
    from huggingface_hub import hf_hub_download, snapshot_download
    from configs.detection.benchmark_config import BENCHMARKS as detection
    from configs.recognition.benchmark_config import BENCHMARKS as recognition

    for name, config in detection.items():
        if config.get("repository") and config.get("filename"):
            announce(name)
            config["model_dir"].mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=config["repository"],
                filename=config["filename"],
                local_dir=config["model_dir"],
            )

    for name, config in recognition.items():
        backend = config.get("backend")
        if backend == "trocr":
            announce(name)
            snapshot_download(repo_id=config["repository"], cache_dir=config["model_dir"])
        elif backend == "turkicocr_onnx":
            announce(name)
            snapshot_download(repo_id=config["repository"], local_dir=config["model_path"].parent)
        elif backend == "tesseract":
            announce(name)
            target = config["traineddata_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file():
                urllib.request.urlretrieve(config["traineddata_url"], target)
        elif backend == "crnn_ctc":
            announce(name)
            for key in ("checkpoint_filename", "config_filename", "symbols_filename"):
                hf_hub_download(
                    repo_id=config["repository"], filename=config[key],
                    local_dir=config["model_dir"],
                )

    announce("EasyOCR CRAFT and Cyrillic recognizers")
    import easyocr

    easy_dir = str(detection["craft_easyocr"]["model_dir"])
    easyocr.Reader(["en", "ru"], gpu=False, model_storage_directory=easy_dir)
    for network in ("cyrillic_g1", "cyrillic_g2"):
        easyocr.Reader(
            ["ru"], gpu=False, detector=False, recognizer=True,
            recog_network=network, model_storage_directory=easy_dir,
        )

    announce("manuscript-ocr detectors and recognizers")
    from manuscript.detectors import EAST, Mask2Former, YOLO
    from manuscript.recognizers import TRBA

    for name in ("east_50_g1",):
        EAST(weights=detection[name]["weights"], device="cpu")
    for name in ("yolo26s_obb_text_g1", "yolo26x_obb_text_g1"):
        YOLO(weights=detection[name]["weights"], device="cpu")
    for name in ("trba_lite_g1", "trba_base_g1"):
        TRBA(weights=recognition[name]["weights"], device="cpu", batch_size=1)
    Mask2Former(weights="mask2former_line_v0_prev", device="cpu")
    gc.collect()

    announce("docTR DB-ResNet50")
    from doctr.models import detection_predictor

    detection_predictor("db_resnet50", pretrained=True)


def download_paddleocr():
    from configs.detection.benchmark_config import BENCHMARKS as detection

    home = detection["paddleocr_v6_medium_word"]["model_dir"]
    home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(home))
    import torch  # noqa: F401
    from paddleocr import PaddleOCR, TextRecognition

    config = detection["paddleocr_v6_medium_word"]
    announce("PaddleOCR PP-OCRv6 detector")
    PaddleOCR(
        text_detection_model_name=config["detection_model"],
        text_recognition_model_name=config["recognition_model"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
        device="cpu",
    )
    for model_name in ("cyrillic_PP-OCRv5_mobile_rec", "eslav_PP-OCRv5_mobile_rec"):
        announce(model_name)
        TextRecognition(model_name=model_name, device="cpu")


def download_rfdetr():
    announce("RF-DETR text-line/text-region models")
    from configs.line_detection.benchmark_config import BENCHMARKS
    from scripts.line_detection.benchmark_rfdetr import download_weights
    for config in BENCHMARKS.values():
        if config.get("backend") == "rfdetr" and config.get("run", False):
            download_weights(config)


def download_ultralytics_line():
    from configs.line_detection.benchmark_config import BENCHMARKS
    from huggingface_hub import hf_hub_download
    for name, config in BENCHMARKS.items():
        if config.get("backend") != "ultralytics" or not config.get("run", False):
            continue
        announce(name)
        config["model_dir"].mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=config["repository"], filename=config["filename"],
            local_dir=config["model_dir"],
        )


def download_doc_ufcn():
    announce("Doc-UFCN generic historical line")
    from scripts.line_detection.benchmark_doc_ufcn import download_model_files
    download_model_files()


def download_surya():
    announce("Surya text-line detector")
    from surya.detection import DetectionPredictor
    DetectionPredictor.local(device="cpu")


def download_kraken():
    announce("Kraken PP-OCRv6 recognition models")
    from configs.recognition.benchmark_config import BENCHMARKS
    from scripts.recognition.benchmark_kraken import MODEL_NAMES, ensure_model
    for name in MODEL_NAMES:
        ensure_model(BENCHMARKS[name])
    # Kraken BLLA is bundled in the installed package.


def download_pero():
    announce("PERO general layout model")
    from scripts.line_detection.benchmark_pero import CONFIG, _download_and_extract
    _download_and_extract(CONFIG)


def download_rtmdet():
    announce("Riksarkivet RTMDet lines")
    from scripts.line_detection.benchmark_rtmdet import CONFIG
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=CONFIG["repository"], local_dir=CONFIG["model_dir"])


DOWNLOADERS = {
    "main": download_main,
    "paddleocr": download_paddleocr,
    "rfdetr": download_rfdetr,
    "ultralytics-line": download_ultralytics_line,
    "doc-ufcn": download_doc_ufcn,
    "surya": download_surya,
    "kraken": download_kraken,
    "pero": download_pero,
    "rtmdet": download_rtmdet,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    args = parser.parse_args()
    DOWNLOADERS[args.group]()
    print(f"\nModel group ready: {args.group}", flush=True)


if __name__ == "__main__":
    main()
