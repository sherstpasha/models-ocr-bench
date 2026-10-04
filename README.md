# Russian OCR Benchmark

Бенчмарк OCR для современного и исторического русского текста. Версия **v1**, дата фиксации — **4 октября 2026 года**.

Инструкция по полному воспроизведению окружений, загрузке моделей и датасетов, запуску задач и пересборке таблиц: [REPRODUCING.md](REPRODUCING.md).

Протестировано **33 модели/конфигурации** на **13 наборах данных** в четырёх задачах: детекция слов, детекция строк, распознавание слов и распознавание строк.

## Модели

| Модель | Источник | Лицензия |
| --- | --- | --- |
| `east_50_g1` | [source](https://github.com/konstantinkozhin/manuscript-ocr) | MIT |
| `yolo26s_obb_text_g1` | [source](https://github.com/konstantinkozhin/manuscript-ocr) | AGPL-3.0 |
| `yolo26x_obb_text_g1` | [source](https://github.com/konstantinkozhin/manuscript-ocr) | AGPL-3.0 |
| `yolo11n_text` | [source](https://huggingface.co/RoyRud1902/yolo11n-text) | Apache-2.0 |
| `yolo11x_dialectic` | [source](https://huggingface.co/Daniil-Domino/yolo11x-dialectic) | AGPL-3.0 |
| `craft_easyocr` | [source](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 |
| `paddleocr_v6_medium_word` | [source](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) | Apache-2.0 |
| `doctr_db_resnet50` | [source](https://github.com/mindee/doctr) | Apache-2.0 |
| `openocr_repvit_db` | [source](https://github.com/Topdu/OpenOCR) | Apache-2.0 |
| `paddleocr_v6_medium_line` | [source](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) | Apache-2.0 |
| `rfdetr_textline_textregion_2xl` | [source](https://huggingface.co/Kansallisarkisto/rfdetr-textline-textregion-detection-2xl) | Apache-2.0 |
| `mask2former_line_v0_prev` | [source](https://github.com/konstantinkozhin/manuscript-ocr) | Apache-2.0 |
| `doc_ufcn_generic_historical_line` | [source](https://huggingface.co/Teklia/doc-ufcn-generic-historical-line) | MIT |
| `surya_text_line_detection` | [source](https://github.com/datalab-to/surya) | Apache-2.0 |
| `kraken_blla_default` | [source](https://github.com/mittagessen/kraken) | Apache-2.0 |
| `pero_layout_general` | [source](https://github.com/DCGM/pero-ocr) | BSD-3-Clause |
| `riksarkivet_rtmdet_lines` | [source](https://huggingface.co/Riksarkivet/rtmdet_lines) | MIT |
| `trba_lite_g1` | [source](https://github.com/konstantinkozhin/manuscript-ocr) | MIT |
| `trba_base_g1` | [source](https://github.com/konstantinkozhin/manuscript-ocr) | MIT |
| `cyrillic_g1` | [source](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 |
| `cyrillic_g2` | [source](https://github.com/JaidedAI/EasyOCR) | Apache-2.0 |
| `paddleocr_cyrillic_v5_mobile` | [source](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec) | Apache-2.0 |
| `paddleocr_eslav_v5_mobile` | [source](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec) | Apache-2.0 |
| `trocr_ru_1700s` | [source](https://huggingface.co/taiga75/ru-trocr-1700s) | MIT |
| `trocr_dialectic_stackmix` | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic-stackmix) | Apache-2.0 |
| `trocr_dialectic` | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic) | Apache-2.0 |
| `trocr_base_ru` | [source](https://huggingface.co/raxtemur/trocr-base-ru) | Apache-2.0 |
| `trocr_base_handwritten_ru` | [source](https://huggingface.co/kazars24/trocr-base-handwritten-ru) | Не указана |
| `cyrillic_large_handwritten` | [source](https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten) | Apache-2.0 |
| `turkicocr_svtrv2_b` | [source](https://huggingface.co/alenisaw/turkicocr-svtrv2-b) | Apache-2.0 |
| `kraken_ppocrv6_medium` | [source](https://zenodo.org/records/21788410) | Apache-2.0 |
| `tesseract_rus_best` | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/rus.traineddata) | Apache-2.0 |
| `tesseract_cyrillic_best` | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/script/Cyrillic.traineddata) | Apache-2.0 |

Лицензии взяты из карточек моделей или исходных репозиториев.

## Датасеты

| Датасет | Источник | Лицензия |
| --- | --- | --- |
| `russian_old_orthography` | [source](https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr) | MIT |
| `YeniseiGovReports-TD` | [source](https://huggingface.co/datasets/anna4uonline/YeniseiGovReports-TD) | MIT |
| `school_notebooks_RU` | [source](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) | MIT |
| `handwritten_essay` | [source](https://huggingface.co/datasets/sherstpasha/handwritten_essay) | CC-BY-NC-3.0 |
| `ICDAR2015` | [source](https://rrc.cvc.uab.es/?ch=4) | Условия ICDAR/RRC |
| `TotalText` | [source](https://github.com/cs-chan/Total-Text-Dataset) | BSD-3-Clause |
| `gota_hovratt_seg` | [source](https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg) | Не указана |
| `svea_hovratt_seg` | [source](https://huggingface.co/datasets/Riksarkivet/svea_hovratt_seg) | Не указана |
| `bergskollegium_relationer_och_skrivelser_seg` | [source](https://huggingface.co/datasets/Riksarkivet/bergskollegium_relationer_och_skrivelser_seg) | Не указана |
| `YeniseiGovReports-HWR` | [source](https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-HWR) | MIT |
| `YeniseiGovReports-PRT` | [source](https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-PRT) | MIT |
| `cyrillic_handwriting` | [source](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset) | CC0-1.0 |
| `DonkeySmallOCR-Numbers-Printed-15random` | [source](https://huggingface.co/datasets/sherstpasha/DonkeySmallOCR-Numbers-Printed-15random) | Не указана |

Лицензии взяты из карточек датасетов или исходных репозиториев.

## Результаты детекции слов

Метрики в ячейке: `F1@0.5 / F1@0.5:0.95 / Dice F1 / Polygon H-mean@0.5`; для всех метрик больше — лучше.

`Mean Metric Rank`: 1 — лучший средний ранг; меньше — лучше.

| Model | Origin | Mean F1@0.5 | Mean F1@0.5:0.95 | Mean Dice F1 | Mean Polygon H-mean@0.5 | Mean Metric Rank | [russian_old_orthography](https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr) | [YeniseiGovReports-TD](https://huggingface.co/datasets/anna4uonline/YeniseiGovReports-TD) | [school_notebooks_RU](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) | [handwritten_essay](https://huggingface.co/datasets/sherstpasha/handwritten_essay) | [ICDAR2015](https://rrc.cvc.uab.es/?ch=4) | [TotalText](https://github.com/cs-chan/Total-Text-Dataset) | Mean GPU FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26x_obb_text_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | **0.8470** | **0.6184** | **0.7858** | **0.7490** | **1.0000** | 0.9321 / 0.6875 / 0.8808 / **0.9333** | 0.8981 / **0.7218** / 0.9236 / 0.9009 | **0.9690** / 0.7998 / **0.7944** / **0.7976** | 0.8934 / 0.6178 / 0.9188 / 0.8954 | **0.6480** / **0.4278** / **0.6579** / **0.5025** | **0.7414** / 0.4558 / 0.5391 / 0.4640 | 7.5668 |
| yolo26s_obb_text_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.8359 | 0.6106 | 0.7801 | 0.7382 | 2.0000 | **0.9322** / **0.6914** / **0.8831** / 0.9328 | 0.8968 / 0.7215 / **0.9239** / 0.8994 | 0.9665 / **0.8049** / 0.7902 / 0.7884 | 0.9135 / 0.6393 / 0.9246 / 0.9147 | 0.5885 / 0.3737 / 0.6261 / 0.4491 | 0.7179 / 0.4325 / 0.5329 / 0.4449 | 10.8463 |
| east_50_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.7626 | 0.4778 | 0.7600 | 0.7267 | 3.5000 | 0.9166 / 0.5819 / 0.8469 / 0.9178 | **0.9015** / 0.6091 / 0.8807 / **0.9053** | 0.9162 / 0.5792 / 0.7657 / 0.7656 | **0.9325** / 0.6531 / 0.9108 / **0.9331** | 0.5354 / 0.2793 / 0.6049 / 0.4542 | 0.3731 / 0.1645 / 0.5513 / 0.3842 | 7.8578 |
| yolo11n_text | [source](https://huggingface.co/RoyRud1902/yolo11n-text) | 0.7428 | 0.4806 | 0.7663 | 0.6891 | 3.5000 | 0.9025 / 0.5370 / 0.8346 / 0.9048 | 0.7018 / 0.4094 / 0.7763 / 0.7069 | 0.7729 / 0.4549 / 0.7872 / 0.6617 | 0.8511 / **0.6631** / **0.9331** / 0.8524 | 0.5258 / 0.3352 / 0.6286 / 0.4637 | 0.7025 / **0.4841** / **0.6379** / **0.5453** | **45.4837** |
| doctr_db_resnet50 | [source](https://github.com/mindee/doctr) | 0.6382 | 0.3186 | 0.6851 | 0.5580 | 5.5000 | 0.8840 / 0.4800 / 0.8194 / 0.8972 | 0.6594 / 0.3037 / 0.7405 / 0.6674 | 0.6543 / 0.3197 / 0.6934 / 0.3800 | 0.7477 / 0.3325 / 0.8480 / 0.7518 | 0.4223 / 0.2252 / 0.5466 / 0.3257 | 0.4617 / 0.2504 / 0.4630 / 0.3257 | 5.6378 |
| craft_easyocr | [source](https://github.com/JaidedAI/EasyOCR) | 0.6165 | 0.3234 | 0.7319 | 0.5133 | 5.5000 | 0.7160 / 0.3330 / 0.7681 / 0.7200 | 0.6236 / 0.3169 / 0.8101 / 0.6268 | 0.6116 / 0.3429 / 0.7442 / 0.4432 | 0.7386 / 0.4481 / 0.8952 / 0.7394 | 0.4027 / 0.2075 / 0.6280 / 0.2532 | 0.6064 / 0.2919 / 0.5455 / 0.2973 | 7.5499 |
| yolo11x_dialectic | [source](https://huggingface.co/Daniil-Domino/yolo11x-dialectic) | 0.5101 | 0.3054 | 0.6177 | 0.4786 | 7.2500 | 0.7988 / 0.4103 / 0.7929 / 0.8069 | 0.4333 / 0.2273 / 0.6369 / 0.4379 | 0.7606 / 0.4679 / 0.7727 / 0.6352 | 0.7482 / 0.5492 / 0.8911 / 0.7489 | 0.0961 / 0.0494 / 0.2856 / 0.0685 | 0.2236 / 0.1283 / 0.3269 / 0.1739 | 38.1433 |
| openocr_repvit_db | [source](https://github.com/Topdu/OpenOCR) | 0.2825 | 0.1368 | 0.5776 | 0.2359 | 8.2500 | 0.0796 / 0.0309 / 0.6892 / 0.0820 | 0.2775 / 0.1158 / 0.6642 / 0.2817 | 0.3981 / 0.1741 / 0.6944 / 0.3436 | 0.1806 / 0.0846 / 0.2994 / 0.1818 | 0.2634 / 0.1425 / 0.5832 / 0.1924 | 0.4956 / 0.2729 / 0.5349 / 0.3337 | 24.1304 |
| paddleocr_v6_medium_word | [source](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) | 0.1913 | 0.0617 | 0.6181 | 0.1738 | 8.5000 | 0.0200 / 0.0064 / 0.7167 / 0.0204 | 0.1832 / 0.0552 / 0.6508 / 0.1891 | 0.1436 / 0.0412 / 0.6209 / 0.0894 | 0.2404 / 0.0713 / 0.7803 / 0.2443 | 0.2328 / 0.0747 / 0.5064 / 0.2188 | 0.3278 / 0.1212 / 0.4337 / 0.2808 | 9.5884 |

## Результаты детекции строк

Метрики в ячейке: `F1@0.5 / F1@0.5:0.95 / Dice F1 / Polygon H-mean@0.5`; для всех метрик больше — лучше.

`Mean Metric Rank`: 1 — лучший средний ранг; меньше — лучше.

| Model | Origin | Mean F1@0.5 | Mean F1@0.5:0.95 | Mean Dice F1 | Mean Polygon H-mean@0.5 | Mean Metric Rank | [russian_old_orthography](https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr) | [handwritten_essay](https://huggingface.co/datasets/sherstpasha/handwritten_essay) | [school_notebooks_ru](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) | [gota_hovratt_seg](https://huggingface.co/datasets/Riksarkivet/gota_hovratt_seg) | [svea_hovratt_seg](https://huggingface.co/datasets/Riksarkivet/svea_hovratt_seg) | [bergskollegium_relationer_och_skrivelser_seg](https://huggingface.co/datasets/Riksarkivet/bergskollegium_relationer_och_skrivelser_seg) | Mean GPU FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mask2former_line_v0_prev | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.9215 | 0.6740 | **0.9075** | 0.8582 | **2.0000** | 0.8977 / 0.6633 / 0.8079 / 0.8977 | 1.0000 / 0.8988 / 0.9821 / 1.0000 | 0.9341 / 0.7529 / 0.9654 / 0.9355 | 0.9204 / 0.6170 / 0.9040 / 0.8447 | 0.8737 / 0.5480 / 0.8881 / 0.8549 | 0.9028 / 0.5639 / 0.8977 / 0.6163 | 0.3985 |
| rfdetr_textline_textregion_2xl | [source](https://huggingface.co/Kansallisarkisto/rfdetr-textline-textregion-detection-2xl) | **0.9636** | **0.7144** | 0.9062 | 0.8218 | 2.2500 | 0.9457 / 0.4541 / 0.8877 / 0.9499 | 0.9827 / 0.7670 / 0.9518 / 0.9797 | 0.9033 / 0.6219 / 0.9078 / 0.8258 | 0.9765 / 0.8469 / 0.9070 / 0.6442 | 0.9917 / 0.8296 / 0.8897 / 0.7597 | 0.9819 / 0.7668 / 0.8929 / 0.7715 | 8.4291 |
| paddleocr_v6_medium_line | [source](https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) | 0.8767 | 0.5611 | 0.8831 | 0.8594 | 3.0000 | 0.9699 / 0.6558 / 0.8990 / 0.9699 | 0.8130 / 0.5307 / 0.9545 / 0.8169 | 0.8156 / 0.5353 / 0.8592 / 0.8061 | 0.8917 / 0.5778 / 0.8658 / 0.8776 | 0.8751 / 0.5273 / 0.8654 / 0.8431 | 0.8950 / 0.5394 / 0.8548 / 0.8429 | 3.6877 |
| kraken_blla_default | [source](https://github.com/mittagessen/kraken) | 0.9195 | 0.4984 | 0.8736 | **0.9170** | 3.2500 | 0.9329 / 0.4310 / 0.9035 / 0.9599 | 0.9489 / 0.6399 / 0.8702 / 0.9273 | 0.8008 / 0.4926 / 0.8646 / 0.7806 | 0.9436 / 0.4787 / 0.8621 / 0.9378 | 0.9180 / 0.4860 / 0.8692 / 0.9204 | 0.9727 / 0.4621 / 0.8721 / 0.9758 | 0.1163 |
| riksarkivet_rtmdet_lines | [source](https://huggingface.co/Riksarkivet/rtmdet_lines) | 0.8214 | 0.4712 | 0.8219 | 0.8489 | 5.2500 | 0.7628 / 0.5563 / 0.7407 / 0.7644 | 0.8716 / 0.5044 / 0.8616 / 0.8604 | 0.8069 / 0.4807 / 0.8441 / 0.7887 | 0.7992 / 0.3432 / 0.8136 / 0.9025 | 0.7279 / 0.3105 / 0.7749 / 0.8169 | 0.9598 / 0.6321 / 0.8962 / 0.9604 | 2.2166 |
| surya_text_line_detection | [source](https://github.com/datalab-to/surya) | 0.7488 | 0.3526 | 0.8823 | 0.7500 | 5.7500 | 0.9119 / 0.5179 / 0.8779 / 0.9166 | 0.9456 / 0.5340 / 0.9165 / 0.9442 | 0.7188 / 0.3804 / 0.8900 / 0.7119 | 0.7622 / 0.3072 / 0.8808 / 0.7512 | 0.3591 / 0.1269 / 0.8545 / 0.3754 | 0.7953 / 0.2493 / 0.8740 / 0.8007 | 2.9921 |
| pero_layout_general | [source](https://github.com/DCGM/pero-ocr) | 0.7432 | 0.3689 | 0.8515 | 0.7435 | 6.5000 | 0.9675 / 0.6716 / 0.9019 / 0.9696 | 0.7687 / 0.2974 / 0.8302 / 0.7648 | 0.5620 / 0.2655 / 0.8309 / 0.5449 | 0.7946 / 0.3646 / 0.8668 / 0.8159 | 0.5620 / 0.2004 / 0.7978 / 0.5864 | 0.8044 / 0.4141 / 0.8813 / 0.7796 | 5.1923 |
| doc_ufcn_generic_historical_line | [source](https://huggingface.co/Teklia/doc-ufcn-generic-historical-line) | 0.2709 | 0.0998 | 0.6032 | 0.2151 | 8.0000 | 0.8470 / 0.4168 / 0.7249 / 0.5701 | 0.0955 / 0.0221 / 0.4866 / 0.0064 | 0.3326 / 0.0896 / 0.5829 / 0.1010 | 0.1937 / 0.0441 / 0.6242 / 0.2048 | 0.0805 / 0.0136 / 0.5370 / 0.0511 | 0.0759 / 0.0126 / 0.6633 / 0.3575 | 4.6750 |

## Результаты распознавания слов

Метрики в ячейке: `Character Similarity / Exact Match / CER / WER`; для первых двух больше — лучше, для CER и WER меньше — лучше.

`Mean Metric Rank`: 1 — лучший средний ранг; меньше — лучше.

| Model | Origin | Mean Character Similarity | Mean Exact Match | Mean CER | Mean WER | Mean Metric Rank | [russian_old_orthography](https://huggingface.co/datasets/nevmenandr/russian-old-orthography-ocr) | [school_notebooks_ru](https://huggingface.co/datasets/ai-forever/school_notebooks_RU) | [handwritten_essay](https://huggingface.co/datasets/sherstpasha/handwritten_essay) | [yenisei_gov_reports_hwr](https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-HWR) | [yenisei_gov_reports_prt](https://huggingface.co/datasets/sherstpasha/YeniseiGovReports-PRT) | [cyrillic_handwriting](https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset) | [donkeysmall_printed_15random](https://huggingface.co/datasets/sherstpasha/DonkeySmallOCR-Numbers-Printed-15random) | Mean words/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trba_base_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | **0.9368** | **0.7618** | **0.0641** | **0.2454** | **1.0000** | **0.9430** / 0.7775 / **0.0587** / 0.2254 | 0.9401 / 0.7976 / **0.0521** / 0.2099 | 0.8768 / 0.6152 / 0.1220 / 0.3977 | **0.9718** / **0.9056** / **0.0283** / **0.1007** | **0.9890** / **0.9649** / **0.0111** / **0.0390** | 0.9407 / 0.6846 / 0.0640 / 0.2923 | 0.8964 / 0.5873 / 0.1125 / 0.4527 | 680.4992 |
| trba_lite_g1 | [source](https://github.com/konstantinkozhin/manuscript-ocr) | 0.9302 | 0.7476 | 0.0715 | 0.2581 | 2.0000 | 0.9318 / 0.7479 / 0.0679 / 0.2544 | **0.9407** / **0.8005** / 0.0527 / **0.2087** | 0.8743 / 0.6246 / 0.1272 / 0.3933 | 0.9636 / 0.8756 / 0.0359 / 0.1305 | 0.9849 / 0.9500 / 0.0157 / 0.0555 | 0.9369 / 0.6794 / 0.0695 / 0.2974 | 0.8794 / 0.5553 / 0.1314 / 0.4667 | 237.8552 |
| trocr_ru_1700s | [source](https://huggingface.co/taiga75/ru-trocr-1700s) | 0.8812 | 0.6752 | 0.1033 | 0.3462 | 3.0000 | 0.9109 / **0.7865** / 0.0610 / **0.2155** | 0.8839 / 0.7171 / 0.0864 / 0.3066 | 0.8785 / 0.6761 / 0.1106 / 0.3554 | 0.7670 / 0.4915 / 0.2106 / 0.5521 | 0.8881 / 0.8088 / 0.0854 / 0.2159 | 0.9268 / 0.6412 / 0.0750 / 0.3267 | 0.9134 / 0.6053 / 0.0940 / 0.4513 | 25.2136 |
| kraken_ppocrv6_medium | [source](https://zenodo.org/records/21788410) | 0.8136 | 0.5385 | 0.1555 | 0.4699 | 4.7500 | 0.7752 / 0.5727 / 0.1442 / 0.4326 | 0.7982 / 0.5386 / 0.1494 / 0.4741 | 0.7593 / 0.4234 / 0.2079 / 0.5930 | 0.7915 / 0.5050 / 0.1928 / 0.5154 | 0.7980 / 0.6336 / 0.1719 / 0.3850 | 0.8399 / 0.3407 / 0.1517 / 0.5706 | 0.9334 / 0.7553 / 0.0708 / 0.3187 | 131.3635 |
| cyrillic_large_handwritten | [source](https://huggingface.co/Kansallisarkisto/cyrillic-large-handwritten) | 0.8245 | 0.5627 | 0.2136 | 0.4773 | 5.2500 | 0.8711 / 0.6159 / 0.1224 / 0.3996 | 0.7657 / 0.4977 / 0.2688 / 0.5567 | 0.6527 / 0.3257 / 0.4482 / 0.7497 | 0.8106 / 0.5472 / 0.2440 / 0.4927 | 0.8745 / 0.6848 / 0.2001 / 0.3721 | 0.8328 / 0.4560 / 0.1753 / 0.5139 | 0.9642 / 0.8120 / 0.0363 / 0.2567 | 8.3004 |
| trocr_dialectic_stackmix | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic-stackmix) | 0.8091 | 0.5389 | 0.1697 | 0.4802 | 5.5000 | 0.6596 / 0.3052 / 0.2644 / 0.6990 | 0.8887 / 0.7343 / 0.0835 / 0.2865 | **0.8899** / **0.7175** / **0.0925** / **0.3027** | 0.7121 / 0.3831 / 0.2697 / 0.6570 | 0.6644 / 0.3360 / 0.3103 / 0.7033 | **0.9424** / **0.6956** / **0.0625** / **0.2774** | 0.9067 / 0.6007 / 0.1051 / 0.4353 | 26.8859 |
| trocr_dialectic | [source](https://huggingface.co/Daniil-Domino/trocr-base-ru-dialectic) | 0.7900 | 0.4931 | 0.1996 | 0.5202 | 7.0000 | 0.5940 / 0.1666 / 0.3561 / 0.8354 | 0.8876 / 0.7256 / 0.0875 / 0.2914 | 0.8808 / 0.6902 / 0.1019 / 0.3266 | 0.7060 / 0.3707 / 0.2862 / 0.6609 | 0.6402 / 0.2705 / 0.3610 / 0.7581 | 0.9340 / 0.6632 / 0.0712 / 0.3113 | 0.8874 / 0.5647 / 0.1332 / 0.4580 | 24.2226 |
| trocr_base_ru | [source](https://huggingface.co/raxtemur/trocr-base-ru) | 0.7883 | 0.4997 | 0.2061 | 0.6001 | 7.5000 | 0.6282 / 0.2702 / 0.3337 / 0.9003 | 0.8630 / 0.6700 / 0.1067 / 0.3667 | 0.8348 / 0.6205 / 0.1512 / 0.4622 | 0.7294 / 0.4082 / 0.2781 / 0.6701 | 0.6292 / 0.2776 / 0.3885 / 0.9544 | 0.9240 / 0.6315 / 0.0816 / 0.3448 | 0.9099 / 0.6200 / 0.1030 / 0.5020 | 25.9984 |
| trocr_base_handwritten_ru | [source](https://huggingface.co/kazars24/trocr-base-handwritten-ru) | 0.7514 | 0.3831 | 0.2355 | 0.6250 | 9.0000 | 0.6337 / 0.1988 / 0.3092 / 0.8063 | 0.8571 / 0.6396 / 0.1182 / 0.3779 | 0.8366 / 0.5338 / 0.1620 / 0.4887 | 0.6463 / 0.2467 / 0.3419 / 0.7749 | 0.6705 / 0.2696 / 0.2855 / 0.7505 | 0.9004 / 0.5272 / 0.1110 / 0.4294 | 0.7155 / 0.2660 / 0.3205 / 0.7473 | 10.5459 |
| paddleocr_cyrillic_v5_mobile | [source](https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec) | 0.4426 | 0.2842 | 0.5551 | 0.7312 | 10.5000 | 0.6858 / 0.3288 / 0.2138 / 0.6724 | 0.0884 / 0.0185 / 0.9157 / 0.9837 | 0.0533 / 0.0051 / 0.9597 / 1.0099 | 0.2421 / 0.1132 / 0.8601 / 0.9153 | 0.8196 / 0.5969 / 0.1505 / 0.4455 | 0.2331 / 0.0408 / 0.7624 / 0.9549 | **0.9759** / **0.8860** / **0.0239** / **0.1367** | 585.6513 |
| turkicocr_svtrv2_b | [source](https://huggingface.co/alenisaw/turkicocr-svtrv2-b) | 0.5784 | 0.1865 | 0.4367 | 0.9587 | 11.2500 | 0.6908 / 0.2125 / 0.2747 / 0.8590 | 0.4191 / 0.0417 / 0.5932 / 1.0813 | 0.3626 / 0.0185 / 0.6766 / 1.1991 | 0.3870 / 0.0758 / 0.6894 / 1.0769 | 0.7517 / 0.3373 / 0.2149 / 0.7517 | 0.5067 / 0.0291 / 0.5360 / 1.0985 | 0.9311 / 0.5907 / 0.0720 / 0.6440 | 471.1562 |
| paddleocr_eslav_v5_mobile | [source](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec) | 0.4235 | 0.2636 | 0.5675 | 0.7457 | 11.7500 | 0.6221 / 0.2639 / 0.2649 / 0.7370 | 0.0840 / 0.0190 / 0.9217 / 0.9829 | 0.0504 / 0.0029 / 0.9576 / 1.0118 | 0.2424 / 0.1179 / 0.8568 / 0.9037 | 0.7641 / 0.5329 / 0.1888 / 0.4849 | 0.2304 / 0.0330 / 0.7560 / 0.9586 | 0.9711 / 0.8753 / 0.0266 / 0.1407 | **699.5498** |
| cyrillic_g2 | [source](https://github.com/JaidedAI/EasyOCR) | 0.4274 | 0.2213 | 0.6409 | 1.1402 | 13.0000 | 0.7473 / 0.3124 / 0.2300 / 0.7457 | 0.1048 / 0.0166 / 0.9654 / 1.3861 | 0.0795 / 0.0037 / 1.0500 / 1.7785 | 0.1622 / 0.0478 / 1.0801 / 1.5249 | 0.7934 / 0.5328 / 0.2032 / 0.6075 | 0.2077 / 0.0078 / 0.8489 / 1.4284 | 0.8970 / 0.6280 / 0.1086 / 0.5100 | 109.6403 |
| cyrillic_g1 | [source](https://github.com/JaidedAI/EasyOCR) | 0.4155 | 0.1638 | 0.6203 | 0.9714 | 14.0000 | 0.5190 / 0.1107 / 0.4861 / 0.9955 | 0.1856 / 0.0191 / 0.8516 / 1.1319 | 0.1161 / 0.0045 / 1.0159 / 1.1587 | 0.1964 / 0.0275 / 0.8571 / 1.0892 | 0.6746 / 0.3318 / 0.3054 / 0.8530 | 0.3184 / 0.0136 / 0.7142 / 1.1389 | 0.8987 / 0.6393 / 0.1118 / 0.4327 | 63.9277 |
| tesseract_cyrillic_best | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/script/Cyrillic.traineddata) | 0.3975 | 0.1859 | 3.1894 | 7.1136 | 14.7500 | 0.8313 / 0.5083 / 0.1755 / 0.5353 | 0.1508 / 0.0108 / 3.8474 / 6.7485 | 0.1763 / 0.0082 / 2.2724 / 4.3560 | 0.1113 / 0.0082 / 5.4663 / 9.7240 | 0.6784 / 0.4250 / 3.4440 / 6.9106 | 0.2754 / 0.0136 / 1.8430 / 3.8638 | 0.5590 / 0.3273 / 5.2769 / 17.6573 | 104.4722 |
| tesseract_rus_best | [source](https://github.com/tesseract-ocr/tessdata_best/blob/main/rus.traineddata) | 0.3756 | 0.1752 | 5.4987 | 11.8931 | 15.7500 | 0.8013 / 0.4633 / 0.7224 / 1.6963 | 0.1360 / 0.0089 / 8.7576 / 15.7970 | 0.1703 / 0.0068 / 4.6738 / 9.1870 | 0.1079 / 0.0069 / 6.8035 / 11.7983 | 0.6767 / 0.4183 / 4.3476 / 8.9314 | 0.1836 / 0.0078 / 7.6413 / 17.2751 | 0.5530 / 0.3147 / 5.5444 / 18.5667 | 84.6194 |

## Результаты распознавания строк

Метрики в ячейке: `Character Similarity / Exact Match / CER / WER`; для первых двух больше — лучше, для CER и WER меньше — лучше.

`Mean Metric Rank`: 1 — лучший средний ранг; меньше — лучше.

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
