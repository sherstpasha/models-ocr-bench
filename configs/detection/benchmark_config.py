"""Everything that normally needs editing before a benchmark run."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results" / "detection"

BENCHMARKS = {
    "east_50_g1": {
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
}

DATASETS = {
    "YeniseiGovReports-TD": {
        "folder": Path(r"C:\benchmark\YeniseiGovReports-TD\test_images"),
        "annotations": Path(r"C:\benchmark\YeniseiGovReports-TD\test.json"),
    },
    "school_notebooks_RU": {
        "folder": Path(
            r"C:\benchmark\school_notebooks_RU\benchmark_validation\images"
        ),
        "annotations": Path(
            r"C:\benchmark\school_notebooks_RU\benchmark_validation\annotations.json"
        ),
    },
    "handwritten_essay": {
        # Mendeley's train/ is the 28-page validation split, but its image names
        # start with test_. COCO flattens paths: train/0/0.png -> test_0_0.png.
        "folder": Path(r"C:\benchmark\handwritten_essay\train"),
        "annotations": Path(r"C:\benchmark\handwritten_essay\train_coco.json"),
        "filename_prefix": "test_",
        "repository": "sherstpasha/handwritten_essay",
        "download_dir": Path(r"C:\benchmark\handwritten_essay"),
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
        "folder": Path(r"C:\benchmark\ICDAR2015\test_images"),
        "annotations": Path(r"C:\benchmark\ICDAR2015\test.json"),
    },
    "TotalText": {
        "folder": Path(r"C:\benchmark\TotalText\test_images"),
        "annotations": Path(r"C:\benchmark\TotalText\test.json"),
    },
}
