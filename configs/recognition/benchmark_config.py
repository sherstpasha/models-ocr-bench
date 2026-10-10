"""Configuration for word- and line-level text recognition benchmarks."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RECOGNITION_LEVEL = os.environ.get("OCR_BENCH_RECOGNITION_LEVEL", "word")
if RECOGNITION_LEVEL not in {"word", "line"}:
    raise ValueError(f"Unsupported recognition level: {RECOGNITION_LEVEL}")
RESULTS_ROOT = PROJECT_ROOT / "benchmark_results" / (
    "line_recognition" if RECOGNITION_LEVEL == "line" else "recognition"
)
DATA_ROOT = Path(os.environ.get(
    "OCR_BENCH_DATA_ROOT",
    r"C:\benchmark" if os.name == "nt" else "benchmark_data",
))
HANDWRITTEN_ESSAY_ROOT = DATA_ROOT / "handwritten_essay"
YENISEI_HWR_ROOT = DATA_ROOT / "YeniseiGovReports-HWR"
YENISEI_PRT_ROOT = DATA_ROOT / "YeniseiGovReports-PRT"
CYRILLIC_HANDWRITING_ROOT = DATA_ROOT / "cyrillic-handwriting-dataset"
DONKEYSMALL_PRINTED_ROOT = DATA_ROOT / "DonkeySmallOCR-Numbers-Printed-15random"
SCHOOL_NOTEBOOKS_ROOT = DATA_ROOT / "school_notebooks_RU"
OLD_ORTHOGRAPHY_ROOT = DATA_ROOT / "russian_old_orthography_ocr"
YENISEI_HWR_DATA = (
    YENISEI_HWR_ROOT / "YeniseiGovReports-HWR" / "YeniseiGovReports-HWR"
)
YENISEI_PRT_DATA = (
    YENISEI_PRT_ROOT / "YeniseiGovReports-PRT" / "YeniseiGovReports-PRT"
)

BENCHMARKS = {
    "trba_lite_g1": {
        "origin": "https://github.com/konstantinkozhin/manuscript-ocr",
        "run": True,
        "backend": "trba",
        "weights": "trba_lite_g1",
        "device": "cuda",
        "batch_size": 128,
        "warmup_size": 128,
        "output_dir": RESULTS_ROOT / "trba_lite_g1",
    },
    "trba_base_g1": {
        "origin": "https://github.com/konstantinkozhin/manuscript-ocr",
        "run": True,
        "backend": "trba",
        "weights": "trba_base_g1",
        "device": "cuda",
        "batch_size": 128,
        "warmup_size": 128,
        "output_dir": RESULTS_ROOT / "trba_base_g1",
    },
    "cyrillic_g1": {
        "origin": "https://github.com/JaidedAI/EasyOCR",
        "run": True,
        "backend": "easyocr",
        "recognition_network": "cyrillic_g1",
        "device": "cuda",
        "warmup_size": 16,
        "model_dir": PROJECT_ROOT / "models" / "easyocr",
        "output_dir": RESULTS_ROOT / "cyrillic_g1",
    },
    "cyrillic_g2": {
        "origin": "https://github.com/JaidedAI/EasyOCR",
        "run": True,
        "backend": "easyocr",
        "recognition_network": "cyrillic_g2",
        "device": "cuda",
        "warmup_size": 16,
        "model_dir": PROJECT_ROOT / "models" / "easyocr",
        "output_dir": RESULTS_ROOT / "cyrillic_g2",
    },
    "paddleocr_cyrillic_v5_mobile": {
        # PaddlePaddle runs in .venv-paddleocr; see README.md.
        "run": False,
        "backend": "paddleocr",
        "origin": "https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec",
        "model_name": "cyrillic_PP-OCRv5_mobile_rec",
        "device": "gpu:0",
        "batch_size": 128,
        "warmup_size": 128,
        "model_dir": PROJECT_ROOT / "models" / "paddleocr",
        "output_dir": RESULTS_ROOT / "paddleocr_cyrillic_v5_mobile",
    },
    "paddleocr_eslav_v5_mobile": {
        # Russian, Belarusian and Ukrainian; runs in .venv-paddleocr.
        "run": False,
        "backend": "paddleocr",
        "origin": "https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec",
        "model_name": "eslav_PP-OCRv5_mobile_rec",
        "device": "gpu:0",
        "batch_size": 128,
        "warmup_size": 128,
        "model_dir": PROJECT_ROOT / "models" / "paddleocr",
        "output_dir": RESULTS_ROOT / "paddleocr_eslav_v5_mobile",
    },
    "trocr_ru_1700s": {
        "origin": "https://huggingface.co/taiga75/ru-trocr-1700s",
        "run": True,
        "backend": "trocr",
        "repository": "taiga75/ru-trocr-1700s",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_ru_1700s",
    },
    "trocr_prereform_orthography": {
        "origin": "https://huggingface.co/Serovvans/trocr-prereform-orthography",
        "run": True,
        "backend": "trocr",
        "repository": "Serovvans/trocr-prereform-orthography",
        "processor_repository": "microsoft/trocr-base-printed",
        "device": "cuda",
        "batch_size": 4,
        "warmup_size": 4,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_prereform_orthography",
    },
    "trocr_russian_18th_century_printed": {
        "origin": "https://huggingface.co/dsmchr/trocr_russian_18th_century_printed",
        # Known issue: native Python/CUDA exits occurred in long runs. Batch 1
        # plus per-batch checkpoints is used for resumable completion.
        "run": True,
        "backend": "trocr",
        "repository": "dsmchr/trocr_russian_18th_century_printed",
        "processor_repository": "microsoft/trocr-base-printed",
        "device": "cuda",
        # Keep sustained decoder load below the level that killed the
        # long-running batch-8 benchmark on Windows.
        "batch_size": 1,
        "warmup_size": 1,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_russian_18th_century_printed",
    },
    "trocr_handwritten_cyrillic": {
        "origin": "https://huggingface.co/cyrillic-trocr/trocr-handwritten-cyrillic",
        "run": True,
        "backend": "trocr",
        "repository": "cyrillic-trocr/trocr-handwritten-cyrillic",
        "device": "cuda",
        # Long generations on Yenisei crops caused a native Python/CUDA exit
        # even at batch 4. Use the conservative single-sample path.
        "batch_size": 1,
        "warmup_size": 1,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_handwritten_cyrillic",
    },
    "trocr_church_slavonic_handwritten": {
        "origin": "https://huggingface.co/cyrillic-trocr/trocr-church-slavonic-handwritten",
        # Known issue: repeated native CUDA/Python exits in long runs. The
        # benchmark uses batch 1 and checkpoints every prediction.
        "run": True,
        "backend": "trocr",
        "repository": "cyrillic-trocr/trocr-church-slavonic-handwritten",
        "tokenizer_json_processor": True,
        "device": "cuda",
        "batch_size": 1,
        "warmup_size": 1,
        "preserve_aspect_height": 128,
        "generation_max_length": 128,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_church_slavonic_handwritten",
    },
    "trocr_dialectic_stackmix": {
        "origin": "https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic-stackmix",
        "run": True,
        "backend": "trocr",
        "repository": "Daniil-Domino/trocr-base-ru-dialectic-stackmix",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_dialectic_stackmix",
    },
    "trocr_dialectic": {
        "origin": "https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic",
        "run": True,
        "backend": "trocr",
        "repository": "Daniil-Domino/trocr-base-ru-dialectic",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_dialectic",
    },
    "trocr_base_ru": {
        "origin": "https://huggingface.co/raxtemur/trocr-base-ru",
        "run": True,
        "backend": "trocr",
        "repository": "raxtemur/trocr-base-ru",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_base_ru",
    },
    "trocr_base_handwritten_ru": {
        "origin": "https://huggingface.co/kazars24/trocr-base-handwritten-ru",
        "run": True,
        "backend": "trocr",
        "repository": "kazars24/trocr-base-handwritten-ru",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_base_handwritten_ru",
    },
    "parseq_s_rukopys": {
        "origin": "https://huggingface.co/Hukyl/parseq-s-rukopys",
        "run": True,
        "backend": "parseq",
        "repository": "Hukyl/parseq-s-rukopys",
        "filename": "best.pt",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "parseq_s_rukopys",
    },
    "parseq_b_rukopys": {
        "origin": "https://huggingface.co/Hukyl/parseq-b-rukopys",
        "run": True,
        "backend": "parseq",
        "repository": "Hukyl/parseq-b-rukopys",
        "filename": "best.pt",
        "device": "cuda",
        "batch_size": 4,
        "warmup_size": 4,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "parseq_b_rukopys",
    },
    "trocr_rukopys": {
        "origin": "https://huggingface.co/Hukyl/trocr-rukopys",
        "run": True,
        "backend": "trocr",
        "repository": "Hukyl/trocr-rukopys",
        "tokenizer_json_processor": True,
        "device": "cuda",
        "batch_size": 1,
        "warmup_size": 1,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_rukopys",
    },
    "trocr_large_rukopys_hw": {
        "origin": "https://huggingface.co/Hukyl/trocr-large-rukopys-hw",
        "run": True,
        "high_power": True,
        "backend": "trocr",
        "repository": "Hukyl/trocr-large-rukopys-hw",
        "tokenizer_json_processor": True,
        "device": "cuda",
        # TrOCR-Large has about 558M parameters; batch 1 keeps the Windows
        # benchmark comfortably resumable on a 16 GB GPU.
        "batch_size": 1,
        "warmup_size": 1,
        "generation_max_length": 512,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "trocr_large_rukopys_hw",
    },
    "cyrillic_htr_model": {
        "origin": "https://huggingface.co/Kansallisarkisto/cyrillic-htr-model",
        "run": True,
        "backend": "trocr",
        "repository": "Kansallisarkisto/cyrillic-htr-model",
        "processor_subfolder": "processor",
        "device": "cuda",
        "batch_size": 8,
        "warmup_size": 8,
        "prediction_postprocess": "character_spaced",
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "cyrillic_htr_model",
    },
    "cyrillic_large_handwritten": {
        "run": True,
        # This large checkpoint sustains the full GPU power limit for a long
        # time. The all-in-one runner skips it unless explicitly requested.
        "high_power": True,
        "backend": "trocr",
        "origin": "https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten",
        "repository": "Kansallisarkisto/cyrillic-large-handwritten",
        "custom_processor": True,
        "device": "cuda",
        # DINOv2-large + ruRoberta-large uses four-beam decoding at 518 px.
        # Batch 1 avoids sustained peak load and VRAM spikes on 16 GB GPUs.
        "batch_size": 1,
        "warmup_size": 1,
        "model_dir": PROJECT_ROOT / "models" / "huggingface",
        "output_dir": RESULTS_ROOT / "cyrillic_large_handwritten",
    },
    "crnn_ctc_church_slavonic": {
        "origin": "https://huggingface.co/achimrabus/crnn-ctc-church-slavonic",
        "run": True,
        "backend": "crnn_ctc",
        "repository": "achimrabus/crnn-ctc-church-slavonic",
        "model_dir": PROJECT_ROOT / "models" / "crnn-ctc-church-slavonic",
        "checkpoint_filename": "best_model.pt",
        "config_filename": "model_config.json",
        "symbols_filename": "symbols.txt",
        "device": "cuda",
        "batch_size": 32,
        "warmup_size": 32,
        "output_dir": RESULTS_ROOT / "crnn_ctc_church_slavonic",
    },
    "party_european_languages": {
        "origin": "https://zenodo.org/records/15764161",
        "repository": "Zenodo 15764161",
        "run": True,
        "backend": "party",
        "download_url": "https://zenodo.org/records/15764161/files/party_european_langs.safetensors?download=1",
        "model_path": PROJECT_ROOT / "models" / "party" / "party_european_langs.safetensors",
        "device": "cuda:0",
        "batch_size": 8,
        # Party accepts a complete page and batches its detected line crops
        # internally. Pack more benchmark crops into each synthetic page to
        # avoid paying page setup overhead once per four words.
        "page_size": 32,
        "warmup_size": 8,
        "output_dir": RESULTS_ROOT / "party_european_languages",
    },
    "turkicocr_svtrv2_b": {
        "run": False,
        "backend": "turkicocr_onnx",
        "origin": "https://huggingface.co/alenisaw/turkicocr-svtrv2-b",
        "repository": "alenisaw/turkicocr-svtrv2-b-onnx",
        "model_path": PROJECT_ROOT / "models" / "turkicocr-svtrv2-b-onnx" / "model.onnx",
        "charset_path": PROJECT_ROOT / "models" / "turkicocr-svtrv2-b-onnx" / "charset_turkic_cyrillic.txt",
        "device": "cuda",
        "batch_size": 128,
        "warmup_size": 128,
        "image_height": 48,
        "max_width": 640,
        "output_dir": RESULTS_ROOT / "turkicocr_svtrv2_b",
    },
    "kraken_ppocrv6_medium": {
        # Multilingual handwritten/printed line recognizer, evaluated here on
        # word crops by treating each crop as a single text line.
        "run": False,
        "backend": "kraken",
        "origin": "https://zenodo.org/records/21788410",
        "repository": "small-models-for-glam/kraken-ppocrv6-medium",
        "filename": "medium.safetensors",
        "model_path": PROJECT_ROOT / "models" / "kraken-ppocrv6-medium" / "medium.safetensors",
        "device": "cuda:0",
        "batch_size": 32,
        "warmup_size": 16,
        "output_dir": RESULTS_ROOT / "kraken_ppocrv6_medium",
    },
    "kraken_ppocrv6_small": {
        "run": False,
        "backend": "kraken",
        "origin": "https://zenodo.org/records/21788405",
        "repository": "small-models-for-glam/kraken-ppocrv6-small",
        "filename": "small.safetensors",
        "model_path": PROJECT_ROOT / "models" / "kraken-ppocrv6-small" / "small.safetensors",
        "device": "cuda:0",
        "batch_size": 32,
        "warmup_size": 16,
        "output_dir": RESULTS_ROOT / "kraken_ppocrv6_small",
    },
    "kraken_ppocrv6_tiny": {
        "run": False,
        "backend": "kraken",
        "origin": "https://zenodo.org/records/21788403",
        "repository": "small-models-for-glam/kraken-ppocrv6-tiny",
        "filename": "tiny.safetensors",
        "model_path": PROJECT_ROOT / "models" / "kraken-ppocrv6-tiny" / "tiny.safetensors",
        "device": "cuda:0",
        "batch_size": 32,
        "warmup_size": 16,
        "output_dir": RESULTS_ROOT / "kraken_ppocrv6_tiny",
    },
    "tesseract_rus_best": {
        "run": False,
        "backend": "tesseract",
        "origin": "https://github.com/tesseract-ocr/tessdata_best/blob/main/rus.traineddata",
        "traineddata_url": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/rus.traineddata",
        "language": "rus_best",
        "traineddata_path": PROJECT_ROOT / "models" / "tesseract" / "tessdata" / "rus_best.traineddata",
        "psm": 8,
        "batch_size": 512,
        "output_dir": RESULTS_ROOT / "tesseract_rus_best",
    },
    "tesseract_cyrillic_best": {
        "run": False,
        "backend": "tesseract",
        "origin": "https://github.com/tesseract-ocr/tessdata_best/blob/main/script/Cyrillic.traineddata",
        "traineddata_url": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/script/Cyrillic.traineddata",
        "language": "Cyrillic_best",
        "traineddata_path": PROJECT_ROOT / "models" / "tesseract" / "tessdata" / "Cyrillic_best.traineddata",
        "psm": 8,
        "batch_size": 512,
        "output_dir": RESULTS_ROOT / "tesseract_cyrillic_best",
    },
}

WORD_DATASETS = {
    "russian_old_orthography": {
        "format": "old_orthography",
        "folder": OLD_ORTHOGRAPHY_ROOT / "word_recognition" / "images",
        "prepared_dir": OLD_ORTHOGRAPHY_ROOT / "word_recognition",
        "images_dir": OLD_ORTHOGRAPHY_ROOT / "word_recognition" / "images",
        "labels": OLD_ORTHOGRAPHY_ROOT / "word_recognition" / "labels.csv",
        "repository": "nevmenandr/russian-old-orthography-ocr",
        "download_dir": OLD_ORTHOGRAPHY_ROOT,
        "download_files": [
            "books-pdf-plaintext/pdf/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.pdf",
            "books-pdf-plaintext/txt/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.txt",
        ],
        "pdf_file": "books-pdf-plaintext/pdf/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.pdf",
        "text_file": "books-pdf-plaintext/txt/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.txt",
    },
    "school_notebooks_ru": {
        "format": "coco_words",
        "folder": SCHOOL_NOTEBOOKS_ROOT / "images",
        "source_annotations": SCHOOL_NOTEBOOKS_ROOT / "annotations_val.json",
        "source_images": SCHOOL_NOTEBOOKS_ROOT / "images",
        "text_categories": ["pupil_text", "pupil_comment", "teacher_comment"],
        "text_attribute": "translation",
        "prepared_dir": SCHOOL_NOTEBOOKS_ROOT / "recognition_validation",
        "images_dir": SCHOOL_NOTEBOOKS_ROOT / "recognition_validation" / "images",
        "labels": SCHOOL_NOTEBOOKS_ROOT / "recognition_validation" / "labels.csv",
        "repository": "ai-forever/school_notebooks_RU",
        "download_dir": SCHOOL_NOTEBOOKS_ROOT,
        "download_files": ["annotations_val.json", "images.zip"],
    },
    "handwritten_essay": {
        "folder": HANDWRITTEN_ESSAY_ROOT / "train",
        "page_annotations": HANDWRITTEN_ESSAY_ROOT / "train_page.json",
        "prepared_dir": HANDWRITTEN_ESSAY_ROOT / "recognition_validation",
        "images_dir": (
            HANDWRITTEN_ESSAY_ROOT / "recognition_validation" / "images"
        ),
        "labels": HANDWRITTEN_ESSAY_ROOT / "recognition_validation" / "labels.csv",
        "repository": "sherstpasha/handwritten_essay",
        "download_dir": HANDWRITTEN_ESSAY_ROOT,
        "download_files": ["train_page.json"],
        "archive_url": (
            "https://data.mendeley.com/public-files/datasets/vs44v8r3nf/files/"
            "3a105c2e-02e9-4fc5-91e2-4d828ceb2d90/file_downloaded"
        ),
        "archive_name": "handwritten_essay.zip",
        "archive_sha256": (
            "274bd7e4be4c63f68d8f39d88ca89414bf30858d27b49731d58cc161481391ff"
        ),
    },
    "yenisei_gov_reports_hwr": {
        "format": "imagefolder",
        "folder": YENISEI_HWR_DATA / "val" / "img",
        "source_labels": YENISEI_HWR_DATA / "val" / "labels.csv",
        "prepared_dir": YENISEI_HWR_ROOT / "recognition_test",
        "images_dir": YENISEI_HWR_ROOT / "recognition_test" / "images",
        "labels": YENISEI_HWR_ROOT / "recognition_test" / "labels.csv",
        "repository": "sherstpasha/YeniseiGovReports-HWR",
        "download_dir": YENISEI_HWR_ROOT,
        "huggingface_archive": "YeniseiGovReports-HWR.rar",
    },
    "yenisei_gov_reports_prt": {
        "format": "imagefolder",
        "folder": YENISEI_PRT_DATA / "val" / "img",
        "source_labels": YENISEI_PRT_DATA / "val" / "labels.csv",
        "prepared_dir": YENISEI_PRT_ROOT / "recognition_test",
        "images_dir": YENISEI_PRT_ROOT / "recognition_test" / "images",
        "labels": YENISEI_PRT_ROOT / "recognition_test" / "labels.csv",
        "repository": "sherstpasha/YeniseiGovReports-PRT",
        "download_dir": YENISEI_PRT_ROOT,
        "huggingface_archive": "YeniseiGovReports-PRT.rar",
    },
    "cyrillic_handwriting": {
        "format": "imagefolder",
        "folder": CYRILLIC_HANDWRITING_ROOT / "test",
        "source_labels": CYRILLIC_HANDWRITING_ROOT / "test.csv",
        "source_delimiter": ",",
        "source_fieldnames": ["filename", "text"],
        "prepared_dir": CYRILLIC_HANDWRITING_ROOT / "recognition_test",
        "images_dir": CYRILLIC_HANDWRITING_ROOT / "recognition_test" / "images",
        "labels": CYRILLIC_HANDWRITING_ROOT / "recognition_test" / "labels.csv",
        "kaggle_dataset": "constantinwerner/cyrillic-handwriting-dataset",
        "source_url": "https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset",
        "kaggle_archive": "cyrillic-handwriting-dataset.zip",
        "download_dir": CYRILLIC_HANDWRITING_ROOT,
    },
    "donkeysmall_printed_15random": {
        "format": "imagefolder",
        "folder": DONKEYSMALL_PRINTED_ROOT / "val" / "val" / "img",
        "source_labels": (
            DONKEYSMALL_PRINTED_ROOT / "val" / "val" / "labels.csv"
        ),
        "source_image_column": "image",
        "prepared_dir": DONKEYSMALL_PRINTED_ROOT / "recognition_test",
        "images_dir": DONKEYSMALL_PRINTED_ROOT / "recognition_test" / "images",
        "labels": DONKEYSMALL_PRINTED_ROOT / "recognition_test" / "labels.csv",
        "repository": "sherstpasha/DonkeySmallOCR-Numbers-Printed-15random",
        "huggingface_snapshot": True,
        "download_dir": DONKEYSMALL_PRINTED_ROOT,
    },
}

LINE_DATASETS = {
    "russian_old_orthography": {
        "format": "old_orthography",
        "folder": OLD_ORTHOGRAPHY_ROOT / "line_recognition" / "images",
        "prepared_dir": OLD_ORTHOGRAPHY_ROOT / "line_recognition",
        "images_dir": OLD_ORTHOGRAPHY_ROOT / "line_recognition" / "images",
        "labels": OLD_ORTHOGRAPHY_ROOT / "line_recognition" / "labels.csv",
        "repository": "nevmenandr/russian-old-orthography-ocr",
        "download_dir": OLD_ORTHOGRAPHY_ROOT,
        "download_files": [
            "books-pdf-plaintext/pdf/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.pdf",
            "books-pdf-plaintext/txt/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.txt",
        ],
        "pdf_file": "books-pdf-plaintext/pdf/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.pdf",
        "text_file": "books-pdf-plaintext/txt/zrazhevskaja_a_v.zhenshchina_poet_i_avtor.txt",
    },
    "school_notebooks_ru": {
        "format": "line_crops",
        "preparation_version": 3,
        "prepare": "school_notebooks_lines",
        "folder": SCHOOL_NOTEBOOKS_ROOT / "benchmark_validation" / "images",
        "source_annotations": SCHOOL_NOTEBOOKS_ROOT / "annotations_val.json",
        "source_images": SCHOOL_NOTEBOOKS_ROOT / "images",
        "line_annotations": SCHOOL_NOTEBOOKS_ROOT / "line_detection_val.json",
        "prepared_dir": SCHOOL_NOTEBOOKS_ROOT / "line_recognition_validation",
        "images_dir": SCHOOL_NOTEBOOKS_ROOT / "line_recognition_validation" / "images",
        "labels": SCHOOL_NOTEBOOKS_ROOT / "line_recognition_validation" / "labels.csv",
        "download_dir": SCHOOL_NOTEBOOKS_ROOT,
        "repository": "ai-forever/school_notebooks_RU",
    },
    "handwritten_essay": {
        "format": "line_crops",
        "preparation_version": 3,
        "prepare": "handwritten_essay_lines",
        "folder": HANDWRITTEN_ESSAY_ROOT / "train",
        "page_annotations": HANDWRITTEN_ESSAY_ROOT / "train_page.json",
        "line_annotations": HANDWRITTEN_ESSAY_ROOT / "line_detection_train.json",
        "prepared_dir": HANDWRITTEN_ESSAY_ROOT / "line_recognition_validation",
        "images_dir": HANDWRITTEN_ESSAY_ROOT / "line_recognition_validation" / "images",
        "labels": HANDWRITTEN_ESSAY_ROOT / "line_recognition_validation" / "labels.csv",
        "download_dir": HANDWRITTEN_ESSAY_ROOT,
        "repository": "sherstpasha/handwritten_essay",
    },
}

DATASETS = LINE_DATASETS if RECOGNITION_LEVEL == "line" else WORD_DATASETS

if RECOGNITION_LEVEL == "line":
    for config in BENCHMARKS.values():
        config["output_dir"] = RESULTS_ROOT / config["output_dir"].name
    # Kraken composes a synthetic page from every batch. Keep a common batch
    # size across the three PP-OCRv6 variants for comparable throughput.
    for config in BENCHMARKS.values():
        if config.get("backend") == "kraken":
            config["batch_size"] = 16
            config["warmup_size"] = 16
    BENCHMARKS["tesseract_rus_best"]["psm"] = 7
    BENCHMARKS["tesseract_cyrillic_best"]["psm"] = 7
