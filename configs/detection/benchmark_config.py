"""Everything that normally needs editing before a benchmark run."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results" / "detection"
DATA_ROOT = Path(os.environ.get(
    "OCR_BENCH_DATA_ROOT",
    r"C:\benchmark" if os.name == "nt" else "benchmark_data",
))
OLD_ORTHOGRAPHY_ROOT = DATA_ROOT / "russian_old_orthography_ocr"

BENCHMARKS = {
    "east_50_yenisei_gov_reports_g1": {
        "origin": "https://github.com/konstantinkozhin/manuscript-ocr",
        "run": True,
        "backend": "manuscript",
        "detector": "east",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_east_50_g1.py",
        "output_dir": RESULTS_ROOT / "east_50_yenisei_gov_reports_g1",
        "weights": "https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_yenisei_gov_reports_g1.onnx",
        "preset": None,
        "target_size": 1408,
        "score_thresh": 0.6,
        "warmup_runs": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "east_50_g1": {
        "origin": "https://github.com/konstantinkozhin/manuscript-ocr",
        "run": True,
        "backend": "manuscript",
        "detector": "east",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_east_50_g1.py",
        "output_dir": RESULTS_ROOT / "east_50_g1",
        "weights": "east_50_g1",
        "preset": None,
        "target_size": 1408,
        "score_thresh": 0.6,
        "warmup_runs": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "yolo26s_obb_text_g1": {
        "origin": "https://github.com/konstantinkozhin/manuscript-ocr",
        "run": True,
        "backend": "manuscript",
        "detector": "yolo",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_east_50_g1.py",
        "output_dir": RESULTS_ROOT / "yolo26s_obb_text_g1",
        "weights": "yolo26s_obb_text_g1",
        "preset": None,
        "target_size": None,
        "score_thresh": 0.1,
        "warmup_runs": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "yolo26x_obb_text_g1": {
        "origin": "https://github.com/konstantinkozhin/manuscript-ocr",
        "run": True,
        "backend": "manuscript",
        "detector": "yolo",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_east_50_g1.py",
        "output_dir": RESULTS_ROOT / "yolo26x_obb_text_g1",
        "weights": "yolo26x_obb_text_g1",
        "preset": None,
        "target_size": None,
        "score_thresh": 0.1,
        "warmup_runs": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "yolo11n_text": {
        "origin": "https://huggingface.co/RoyRud1902/yolo11n-text",
        "run": True,
        "backend": "ultralytics",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_yolo.py",
        "output_dir": RESULTS_ROOT / "yolo11n_text",
        "repository": "RoyRud1902/yolo11n-text",
        "filename": "best.pt",
        "model_dir": PROJECT_ROOT / "models" / "yolo11n-text",
        "imgsz": 640,
        "conf": 0.25,
        "warmup": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "yolo11x_dialectic": {
        "origin": "https://huggingface.co/Daniil-Domino/yolo11x-dialectic",
        "run": True,
        "backend": "ultralytics",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_yolo.py",
        "output_dir": RESULTS_ROOT / "yolo11x_dialectic",
        "repository": "Daniil-Domino/yolo11x-dialectic",
        "filename": "model.pt",
        "model_dir": PROJECT_ROOT / "models" / "yolo11x-dialectic",
        "imgsz": 640,
        "conf": 0.3,
        "warmup": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "craft_easyocr": {
        "origin": "https://github.com/JaidedAI/EasyOCR",
        "run": True,
        "backend": "easyocr",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_craft_easyocr.py",
        "output_dir": RESULTS_ROOT / "craft_easyocr",
        "model_dir": PROJECT_ROOT / "models" / "easyocr",
        "canvas_size": 2560,
        "mag_ratio": 1.0,
        "text_threshold": 0.7,
        "low_text": 0.4,
        "link_threshold": 0.4,
        "warmup": 3,
        "cpu_only": False,
        "gpu_only": True,
    },
    "paddleocr_v6_medium_word": {
        "origin": "https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html",
        # PaddlePaddle and PyTorch require conflicting CUDA DLL sets on Windows.
        # Run this backend from its isolated environment; see README.md.
        "run": False,
        "backend": "paddleocr",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_paddleocr.py",
        "output_dir": RESULTS_ROOT / "paddleocr_v6_medium_word",
        "model_dir": PROJECT_ROOT / "models" / "paddleocr",
        "detection_model": "PP-OCRv6_medium_det",
        "recognition_model": "cyrillic_PP-OCRv5_mobile_rec",
        "device": "gpu:0",
        "return_word_box": True,
        "text_det_limit_side_len": 2560,
        "text_det_limit_type": "max",
        "warmup": 1,
        "cpu_only": False,
        "gpu_only": True,
    },
    "doctr_db_resnet50": {
        "origin": "https://github.com/mindee/doctr",
        "run": True,
        "backend": "doctr",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_doctr.py",
        "output_dir": RESULTS_ROOT / "doctr_db_resnet50",
        "architecture": "db_resnet50",
        "assume_straight_pages": True,
        "preserve_aspect_ratio": True,
        "symmetric_pad": True,
        "warmup": 1,
        "cpu_only": False,
        "gpu_only": True,
    },
    "openocr_repvit_db": {
        "origin": "https://github.com/Topdu/OpenOCR",
        "run": True,
        "backend": "openocr",
        "script": PROJECT_ROOT / "scripts" / "detection" / "benchmark_openocr.py",
        "output_dir": RESULTS_ROOT / "openocr_repvit_db",
        "model_dir": PROJECT_ROOT / "models" / "openocr",
        "repository": "topdu/OpenOCR",
        "filename": "openocr_det_model.onnx",
        "det_input_size": 960,
        "warmup": 1,
        "cpu_only": False,
        "gpu_only": True,
    },
}

DATASETS = {
    "russian_old_orthography": {
        "folder": OLD_ORTHOGRAPHY_ROOT / "pages",
        "annotations": OLD_ORTHOGRAPHY_ROOT / "word_detection.json",
        "repository": "nevmenandr/russian-old-orthography-ocr",
        "download_dir": OLD_ORTHOGRAPHY_ROOT,
        "download_files": [
            "books-pdf-plaintext/pdf/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.pdf",
            "books-pdf-plaintext/txt/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.txt",
        ],
        "pdf_file": "books-pdf-plaintext/pdf/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.pdf",
        "text_file": "books-pdf-plaintext/txt/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.txt",
        "prepare": "old_orthography",
    },
    "YeniseiGovReports-TD": {
        "folder": DATA_ROOT / "YeniseiGovReports-TD" / "test_images",
        "annotations": DATA_ROOT / "YeniseiGovReports-TD" / "test.json",
        "repository": "anna4uonline/YeniseiGovReports-TD",
        "download_dir": DATA_ROOT / "YeniseiGovReports-TD",
        "download_files": ["test.json", "test_images.zip"],
    },
    "school_notebooks_RU": {
        "folder": (
            DATA_ROOT / "school_notebooks_RU" / "benchmark_validation" / "images"
        ),
        "annotations": (
            DATA_ROOT
            / "school_notebooks_RU"
            / "benchmark_validation"
            / "annotations.json"
        ),
        "repository": "ai-forever/school_notebooks_RU",
        "download_dir": DATA_ROOT / "school_notebooks_RU",
        "download_files": ["annotations_val.json", "images.zip"],
        "source_images": DATA_ROOT / "school_notebooks_RU" / "images",
        "source_annotations": (
            DATA_ROOT / "school_notebooks_RU" / "annotations_val.json"
        ),
        "prepare": "school_notebooks",
    },
    "handwritten_essay": {
        # Mendeley's train/ is the 28-page validation split, but its image names
        # start with test_. COCO flattens paths: train/0/0.png -> test_0_0.png.
        "folder": DATA_ROOT / "handwritten_essay" / "train",
        "annotations": DATA_ROOT / "handwritten_essay" / "train_coco.json",
        "filename_prefix": "test_",
        "repository": "sherstpasha/handwritten_essay",
        "download_dir": DATA_ROOT / "handwritten_essay",
        "download_files": [
            "README.md",
            "example_word_annotations.jpg",
            "train_page.json",
            "test_page.json",
            "train_coco.json",
            "test_coco.json",
        ],
        "archive_url": (
            "https://data.mendeley.com/public-files/datasets/vs44v8r3nf/files/"
            "3a105c2e-02e9-4fc5-91e2-4d828ceb2d90/file_downloaded"
        ),
        "archive_name": "handwritten_essay.zip",
        "archive_sha256": (
            "274bd7e4be4c63f68d8f39d88ca89414bf30858d27b49731d58cc161481391ff"
        ),
    },
    "ICDAR2015": {
        "folder": DATA_ROOT / "ICDAR2015" / "test_images",
        "annotations": DATA_ROOT / "ICDAR2015" / "test.json",
        "manual": True,
        "source_url": "https://rrc.cvc.uab.es/?ch=4",
    },
    "TotalText": {
        "folder": DATA_ROOT / "TotalText" / "test_images",
        "annotations": DATA_ROOT / "TotalText" / "test.json",
        "manual": True,
        "source_url": "https://github.com/cs-chan/Total-Text-Dataset",
    },
}
