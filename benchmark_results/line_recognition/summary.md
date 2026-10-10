# Line recognition benchmark summary

Each dataset cell: `Character Similarity / Exact Match / CER / WER`.
`*` marks a dataset known to have been used to train that model.
`Mean lines/s` is the arithmetic mean across available datasets on the device used by each model.

## Aggregate metrics

Character Similarity is `1 - edit_distance / max(reference_length, prediction_length)` averaged across samples.
Exact Match is the share of fully correct samples. Character Similarity and Exact Match range from 0 to 1; higher is better.
CER and WER are normalized edit errors; lower is better and values can exceed 1 when predictions contain many insertions.
Mean Metric Rank averages the ranks of all four mean metrics. Rank 1 is best; rows are sorted by this rank in ascending order.

| Model | Origin | Mean Character Similarity | Mean Exact Match | Mean CER | Mean WER | Mean Metric Rank | [russian_old_orthography](https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr) | [school_notebooks_ru](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) | [handwritten_essay](https://huggingface.co/datasets/sherstpasha/handwritten_essay) | Mean lines/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kraken_ppocrv6_medium | [source](https://zenodo.org/records/21788410) | **0.88** | 0.16 | **0.10** | 0.38 | **1.75** | 0.95 / 0.25 / **0.05** / 0.23 | 0.84 / 0.21 / **0.09** / 0.39 | **0.85** / 0.03 / **0.15** / 0.53 | 47.66 |
| cyrillic_large_handwritten | [source](https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten) | 0.87 | 0.18 | 0.12 | **0.36** | **1.75** | **0.95** / 0.18 / 0.05 / 0.24 | 0.85 / 0.29 / 0.10 / **0.35** | 0.80 / **0.07** / 0.19 / **0.48** | 4.65 |
| trocr_ru_1700s | [source](https://huggingface.co/taiga75/ru-trocr-1700s) | 0.85 | **0.18** | 0.15 | 0.41 | 2.75 | 0.95 / 0.25 / 0.05 / 0.21 | 0.85 / 0.25 / 0.13 / 0.44 | 0.76 / 0.04 / 0.26 / 0.57 | 9.14 |
| kraken_ppocrv6_small | [source](https://zenodo.org/records/21788405) | 0.85 | 0.11 | 0.13 | 0.47 | 4.00 | 0.93 / 0.16 / 0.07 / 0.29 | 0.81 / 0.16 / 0.12 / 0.48 | 0.80 / 0.01 / 0.19 / 0.65 | 72.06 |
| parseq_s_rukopys | [source](https://huggingface.co/Hukyl/parseq-s-rukopys) | 0.84 | 0.11 | 0.17 | 0.56 | 6.75 | 0.82 / 0.01 / 0.18 / 0.58 | **0.89** / 0.29 / 0.11 / 0.43 \* | 0.80 / 0.02 / 0.22 / 0.66 | 58.44 |
| party_european_languages | [source](https://zenodo.org/records/15764161) | 0.76 | 0.13 | 0.21 | 0.49 | 6.75 | 0.93 / **0.28** / 0.07 / **0.21** | 0.69 / 0.10 / 0.25 / 0.58 | 0.68 / 0.01 / 0.32 / 0.68 | 1.24 |
| parseq_b_rukopys | [source](https://huggingface.co/Hukyl/parseq-b-rukopys) | 0.83 | 0.11 | 0.19 | 0.58 | 7.50 | 0.80 / 0.00 / 0.22 / 0.61 | 0.89 / **0.30** / 0.12 / 0.45 \* | 0.79 / 0.02 / 0.24 / 0.68 | 28.17 |
| trocr_handwritten_cyrillic | [source](https://huggingface.co/cyrillic-trocr/trocr-handwritten-cyrillic) | 0.80 | 0.07 | 0.19 | 0.55 | 8.25 | 0.91 / 0.09 / 0.09 / 0.36 | 0.77 / 0.12 / 0.19 / 0.59 | 0.73 / 0.01 / 0.28 / 0.71 | 2.09 |
| trocr_large_rukopys_hw | [source](https://huggingface.co/Hukyl/trocr-large-rukopys-hw) | 0.84 | 0.02 | 0.16 | 0.55 | 8.50 | 0.84 / 0.02 / 0.16 / 0.55 | - | - | 1.73 |
| cyrillic_htr_model | [source](https://huggingface.co/Kansallisarkisto/cyrillic-htr-model) | 0.75 | 0.04 | 0.25 | 0.67 | 11.25 | 0.88 / 0.03 / 0.12 / 0.47 | 0.73 / 0.09 / 0.24 / 0.68 | 0.63 / 0.00 / 0.38 / 0.85 | 1.67 |
| kraken_ppocrv6_tiny | [source](https://zenodo.org/records/21788403) | 0.74 | 0.03 | 0.23 | 0.68 | 12.25 | 0.87 / 0.03 / 0.12 / 0.46 | 0.70 / 0.06 / 0.23 / 0.75 | 0.66 / 0.00 / 0.33 / 0.84 | 83.50 |
| trocr_dialectic_stackmix | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic-stackmix) | 0.70 | 0.08 | 0.31 | 0.81 | 12.50 | 0.63 / 0.01 / 0.38 / 0.93 | 0.81 / 0.21 / 0.19 / 0.66 | 0.66 / 0.02 / 0.37 / 0.85 | 8.30 |
| trba_lite_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.61 | 0.10 | 0.43 | 0.79 | 13.25 | 0.62 / 0.04 / 0.41 / 0.81 | 0.79 / 0.26 / 0.27 / 0.61 \* | 0.43 / 0.01 / 0.61 / 0.95 | **669.31** |
| trocr_rukopys | [source](https://huggingface.co/Hukyl/trocr-rukopys) | 0.69 | 0.03 | 0.33 | 0.80 | 14.25 | 0.73 / 0.00 / 0.28 / 0.79 | 0.72 / 0.09 / 0.30 / 0.75 | 0.61 / 0.00 / 0.41 / 0.85 | 1.98 |
| trocr_base_ru | [source](https://huggingface.co/raxtemur/trocr-base-ru) | 0.69 | 0.06 | 0.32 | 0.87 | 14.75 | 0.67 / 0.01 / 0.34 / 0.91 | 0.78 / 0.16 / 0.22 / 0.77 | 0.63 / 0.02 / 0.40 / 0.93 | 8.20 |
| trocr_dialectic | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic) | 0.67 | 0.07 | 0.35 | 0.85 | 15.25 | 0.56 / 0.00 / 0.45 / 0.96 | 0.80 / 0.19 / 0.20 / 0.71 | 0.64 / 0.02 / 0.39 / 0.89 | 8.19 |
| trba_base_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.60 | 0.10 | 0.44 | 0.82 | 15.25 | 0.63 / 0.04 / 0.40 / 0.83 | 0.78 / 0.24 / 0.29 / 0.68 \* | 0.39 / 0.01 / 0.64 / 0.96 | 523.50 |
| trocr_russian_18th_century_printed | [source](https://huggingface.co/dsmchr/trocr_russian_18th_century_printed) | 0.50 | 0.02 | 0.50 | 0.80 | 18.00 | 0.89 / 0.06 / 0.11 / 0.45 | 0.34 / 0.01 / 0.67 / 0.97 | 0.28 / 0.00 / 0.73 / 0.99 | 2.32 |
| paddleocr_eslav_v5_mobile | [source](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec) | 0.47 | 0.01 | 0.50 | 0.81 | 19.25 | 0.87 / 0.02 / 0.12 / 0.49 | 0.20 / 0.01 / 0.75 / 0.99 | 0.34 / 0.00 / 0.63 / 0.95 | 276.64 |
| trocr_base_handwritten_ru | [source](https://huggingface.co/kazars24/trocr-base-handwritten-ru) | 0.49 | 0.04 | 0.55 | 0.93 | 19.75 | 0.46 / 0.00 / 0.57 / 0.96 | 0.62 / 0.11 / 0.44 / 0.85 | 0.39 / 0.00 / 0.64 / 0.98 | 2.82 |
| turkicocr_svtrv2_b | [source](https://huggingface.co/alenisaw/turkicocr-svtrv2-b) | 0.59 | 0.00 | 0.40 | 0.90 | 20.25 | 0.80 / 0.00 / 0.20 / 0.76 | 0.50 / 0.01 / 0.47 / 0.96 | 0.48 / 0.00 / 0.53 / 0.98 | 34.91 |
| paddleocr_cyrillic_v5_mobile | [source](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec) | 0.43 | 0.01 | 0.55 | 0.83 | 21.25 | 0.88 / 0.02 / 0.12 / 0.51 | 0.20 / 0.01 / 0.77 / 0.99 | 0.22 / 0.00 / 0.76 / 0.98 | 269.51 |
| trocr_prereform_orthography | [source](https://huggingface.co/Serovvans/trocr-prereform-orthography) | 0.41 | 0.02 | 0.63 | 0.88 | 22.00 | 0.88 / 0.05 / 0.12 / 0.50 | 0.18 / 0.00 / 0.90 / 1.12 | 0.16 / 0.00 / 0.85 / 1.03 | 7.41 |
| cyrillic_g1 | [source](https://github.com/JaidedAI/EasyOCR) | 0.40 | 0.00 | 0.59 | 0.97 | 25.00 | 0.75 / 0.00 / 0.25 / 0.79 | 0.25 / 0.00 / 0.73 / 1.06 | 0.22 / 0.00 / 0.79 / 1.07 | 16.43 |
| trocr_church_slavonic_handwritten | [source](https://huggingface.co/cyrillic-trocr/trocr-church-slavonic-handwritten) | 0.36 | 0.00 | 0.66 | 1.10 | 25.50 | 0.60 / 0.00 / 0.42 / 1.08 | 0.26 / 0.00 / 0.79 / 1.17 | 0.23 / 0.00 / 0.77 / 1.04 | 3.43 |
| tesseract_cyrillic_best | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/script/Cyrillic.traineddata) | 0.31 | 0.01 | 1.89 | 3.30 | 26.25 | 0.73 / 0.03 / 1.80 / 3.58 | 0.09 / 0.00 / 1.27 / 1.74 | 0.11 / 0.00 / 2.61 / 4.57 | 32.33 |
| cyrillic_g2 | [source](https://github.com/JaidedAI/EasyOCR) | 0.36 | 0.00 | 0.67 | 1.19 | 26.75 | 0.73 / 0.00 / 0.30 / 0.76 | 0.17 / 0.00 / 0.83 / 1.24 | 0.18 / 0.00 / 0.86 / 1.58 | 26.49 |
| crnn_ctc_church_slavonic | [source](https://huggingface.co/achimrabus/crnn-ctc-church-slavonic) | 0.23 | 0.00 | 0.76 | 0.98 | 27.00 | 0.57 / 0.00 / 0.42 / 0.95 | 0.04 / 0.00 / 0.95 / 1.00 | 0.08 / 0.00 / 0.91 / 1.00 | 155.32 |
| tesseract_rus_best | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/rus.traineddata) | 0.32 | 0.01 | 2.48 | 4.42 | 27.25 | 0.80 / 0.02 / 0.92 / 1.97 | 0.08 / 0.00 / 2.59 / 4.27 | 0.09 / 0.00 / 3.94 / 7.03 | 25.36 |
