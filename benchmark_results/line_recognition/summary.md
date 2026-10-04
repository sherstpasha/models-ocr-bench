# Line recognition benchmark summary

Each dataset cell: `Character Similarity / Exact Match / CER / WER`.
`Mean lines/s` is the arithmetic mean across available datasets on the device used by each model.

## Aggregate metrics

Character Similarity is `1 - edit_distance / max(reference_length, prediction_length)` averaged across samples.
Exact Match is the share of fully correct samples. Character Similarity and Exact Match range from 0 to 1; higher is better.
CER and WER are normalized edit errors; lower is better and values can exceed 1 when predictions contain many insertions.
Mean Metric Rank averages the ranks of all four mean metrics. Rank 1 is best; rows are sorted by this rank in ascending order.

| Model | Origin | Mean Character Similarity | Mean Exact Match | Mean CER | Mean WER | Mean Metric Rank | [russian_old_orthography](https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr) | [school_notebooks_ru](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) | [handwritten_essay](https://huggingface.co/datasets/sherstpasha/handwritten_essay) | Mean lines/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kraken_ppocrv6_medium | [source](https://zenodo.org/records/21788410) | **0.8675** | 0.1515 | **0.1020** | 0.3961 | **1.7500** | 0.9466 / 0.2489 / **0.0501** / 0.2256 | 0.8166 / 0.1835 / **0.0986** / 0.4031 | **0.8394** / 0.0222 / **0.1573** / 0.5595 | 42.3128 |
| cyrillic_large_handwritten | [source](https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten) | 0.8659 | 0.1773 | 0.1152 | **0.3589** | **1.7500** | **0.9467** / 0.1792 / 0.0516 / 0.2445 | 0.8470 / **0.2862** / 0.1009 / **0.3509** | 0.8040 / **0.0666** / 0.1932 / **0.4813** | 4.6476 |
| trocr_ru_1700s | [source](https://huggingface.co/taiga75/ru-trocr-1700s) | 0.8534 | **0.1805** | 0.1460 | 0.4087 | 2.5000 | 0.9456 / **0.2525** / 0.0542 / **0.2139** | **0.8501** / 0.2504 / 0.1270 / 0.4389 | 0.7646 / 0.0385 / 0.2568 / 0.5733 | 9.1444 |
| trocr_dialectic_stackmix | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic-stackmix) | 0.7008 | 0.0782 | 0.3122 | 0.8107 | 5.0000 | 0.6309 / 0.0071 / 0.3819 / 0.9266 | 0.8112 / 0.2052 / 0.1851 / 0.6564 | 0.6603 / 0.0222 / 0.3695 / 0.8492 | 8.2973 |
| trba_lite_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.6130 | 0.1024 | 0.4284 | 0.7902 | 5.7500 | 0.6215 / 0.0356 / 0.4074 / 0.8132 | 0.7922 / 0.2627 / 0.2708 / 0.6117 | 0.4253 / 0.0089 / 0.6070 / 0.9456 | **669.3085** |
| trocr_base_ru | [source](https://huggingface.co/raxtemur/trocr-base-ru) | 0.6937 | 0.0616 | 0.3196 | 0.8671 | 7.0000 | 0.6674 / 0.0085 / 0.3450 / 0.9059 | 0.7823 / 0.1584 / 0.2172 / 0.7653 | 0.6316 / 0.0178 / 0.3967 / 0.9302 | 8.1993 |
| trocr_dialectic | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic) | 0.6684 | 0.0711 | 0.3454 | 0.8534 | 7.0000 | 0.5618 / 0.0050 / 0.4509 / 0.9595 | 0.8008 / 0.1920 / 0.1979 / 0.7115 | 0.6425 / 0.0163 / 0.3876 / 0.8892 | 8.1864 |
| trba_base_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.6009 | 0.1008 | 0.4439 | 0.8199 | 7.2500 | 0.6301 / 0.0441 / 0.4014 / 0.8256 | 0.7826 / 0.2448 / 0.2866 / 0.6752 | 0.3899 / 0.0133 / 0.6437 / 0.9590 | 523.5029 |
| paddleocr_eslav_v5_mobile | [source](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec) | 0.4710 | 0.0091 | 0.5008 | 0.8083 | 9.2500 | 0.8729 / 0.0220 / 0.1154 / 0.4921 | 0.2040 / 0.0051 / 0.7532 / 0.9863 | 0.3362 / 0.0000 / 0.6339 / 0.9465 | 276.6431 |
| turkicocr_svtrv2_b | [source](https://huggingface.co/alenisaw/turkicocr-svtrv2-b) | 0.5917 | 0.0047 | 0.4015 | 0.8997 | 10.2500 | 0.8026 / 0.0043 / 0.2009 / 0.7588 | 0.4970 / 0.0098 / 0.4719 / 0.9578 | 0.4756 / 0.0000 / 0.5316 / 0.9826 | 34.9120 |
| trocr_base_handwritten_ru | [source](https://huggingface.co/kazars24/trocr-base-handwritten-ru) | 0.4872 | 0.0398 | 0.5484 | 0.9273 | 10.5000 | 0.4551 / 0.0036 / 0.5675 / 0.9599 | 0.6180 / 0.1128 / 0.4410 / 0.8459 | 0.3883 / 0.0030 / 0.6367 / 0.9761 | 2.8161 |
| paddleocr_cyrillic_v5_mobile | [source](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec) | 0.4340 | 0.0089 | 0.5509 | 0.8253 | 11.0000 | 0.8796 / 0.0185 / 0.1151 / 0.5091 | 0.1986 / 0.0082 / 0.7727 / 0.9874 | 0.2239 / 0.0000 / 0.7648 / 0.9793 | 269.5086 |
| cyrillic_g1 | [source](https://github.com/JaidedAI/EasyOCR) | 0.4029 | 0.0006 | 0.5906 | 0.9718 | 13.7500 | 0.7471 / 0.0000 / 0.2536 / 0.7880 | 0.2453 / 0.0017 / 0.7310 / 1.0567 | 0.2163 / 0.0000 / 0.7872 / 1.0707 | 16.4260 |
| tesseract_cyrillic_best | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/script/Cyrillic.traineddata) | 0.3124 | 0.0103 | 1.8913 | 3.2957 | 14.0000 | 0.7340 / 0.0299 / 1.8003 / 3.5768 | 0.0889 / 0.0011 / 1.2672 / 1.7371 | 0.1144 / 0.0000 / 2.6064 / 4.5733 | 32.3277 |
| cyrillic_g2 | [source](https://github.com/JaidedAI/EasyOCR) | 0.3586 | 0.0009 | 0.6663 | 1.1904 | 14.2500 | 0.7302 / 0.0007 / 0.3019 / 0.7588 | 0.1693 / 0.0019 / 0.8319 / 1.2361 | 0.1762 / 0.0000 / 0.8650 / 1.5763 | 26.4851 |
| tesseract_rus_best | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/rus.traineddata) | 0.3241 | 0.0068 | 2.4826 | 4.4226 | 15.0000 | 0.8043 / 0.0199 / 0.9157 / 1.9673 | 0.0789 / 0.0006 / 2.5924 / 4.2709 | 0.0890 / 0.0000 / 3.9398 / 7.0296 | 25.3625 |
