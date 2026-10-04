"""Prepare one small annotated PDF for all OCR benchmark levels.

The PDF contains a positioned OCR text layer.  Its coordinates provide the
geometry; the paired, human-readable TXT provides the reference spelling.
"""

import csv
import json
import math
import os
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

PREPARATION_VERSION = 4
BOOK_ID = "zrazhevskaja_a_v.zhenshchina_poet_i_avtor"
BOOK_TITLE = "А. В. Зражевская — «Женщина — поэт и автор»"
RENDER_SCALE = 2.0


def _normal_char(character):
    character = character.lower().replace("ё", "е")
    return character if character.isalnum() else ""


def _reference_tokens(text):
    tokens = []
    normalized = []
    owner = []
    for match in re.finditer(r"\S+", text):
        token = match.group(0).strip()
        clean = "".join(_normal_char(char) for char in token)
        if not clean:
            continue
        token_id = len(tokens)
        tokens.append(token)
        normalized.extend(clean)
        owner.extend([token_id] * len(clean))
    return tokens, "".join(normalized), owner


def _pdf_words(text_page):
    words = []
    current = []

    def finish():
        nonlocal current
        visible = [(char, box) for char, box in current if box is not None]
        if visible:
            xs0 = [box[0] for _char, box in visible]
            ys0 = [box[1] for _char, box in visible]
            xs1 = [box[2] for _char, box in visible]
            ys1 = [box[3] for _char, box in visible]
            words.append(
                {
                    "text": "".join(char for char, _box in current),
                    "pdf_box": (min(xs0), min(ys0), max(xs1), max(ys1)),
                }
            )
        current = []

    for index in range(text_page.count_chars()):
        character = text_page.get_text_range(index, 1)
        if "\r" in character or "\n" in character:
            finish()
            continue
        if character.isspace():
            finish()
            continue
        try:
            box = text_page.get_charbox(index)
            if box[2] <= box[0] or box[3] <= box[1]:
                box = None
        except Exception:
            box = None
        if box is not None:
            previous_boxes = [previous for _char, previous in current if previous is not None]
            if previous_boxes:
                previous = previous_boxes[-1]
                center = (box[1] + box[3]) / 2
                previous_center = (previous[1] + previous[3]) / 2
                height = max(box[3] - box[1], previous[3] - previous[1])
                if abs(center - previous_center) > max(2.0, 0.65 * height):
                    finish()
        # PDFium uses U+FFFE for a discretionary hyphen in these files.
        current.append(("-" if character == "\ufffe" else character, box))
    finish()
    _assign_geometric_lines(words)
    return words


def _assign_geometric_lines(words):
    """Cluster positioned PDF words into visual lines by vertical overlap."""
    clusters = []
    ordered = sorted(
        enumerate(words),
        key=lambda item: (
            -(item[1]["pdf_box"][1] + item[1]["pdf_box"][3]) / 2,
            item[1]["pdf_box"][0],
        ),
    )
    for word_id, word in ordered:
        left, bottom, right, top = word["pdf_box"]
        center = (bottom + top) / 2
        height = top - bottom
        candidates = [
            (abs(center - cluster["center"]), cluster)
            for cluster in clusters
            if abs(center - cluster["center"])
            <= max(2.0, 0.65 * max(height, cluster["height"]))
        ]
        if candidates:
            cluster = min(candidates, key=lambda item: item[0])[1]
            cluster["members"].append(word_id)
            boxes = [words[index]["pdf_box"] for index in cluster["members"]]
            cluster["center"] = sum((box[1] + box[3]) / 2 for box in boxes) / len(boxes)
            cluster["height"] = sum(box[3] - box[1] for box in boxes) / len(boxes)
        else:
            clusters.append({"center": center, "height": height, "members": [word_id]})
    clusters.sort(key=lambda cluster: -cluster["center"])
    for line_id, cluster in enumerate(clusters):
        for word_id in cluster["members"]:
            words[word_id]["line"] = line_id


def _align_words(words, reference_text):
    tokens, reference, reference_owner = _reference_tokens(reference_text)
    pdf_chars = []
    pdf_owner = []
    for word_id, word in enumerate(words):
        clean = "".join(_normal_char(char) for char in word["text"])
        pdf_chars.extend(clean)
        pdf_owner.extend([word_id] * len(clean))

    word_tokens = {index: set() for index in range(len(words))}
    matcher = SequenceMatcher(None, "".join(pdf_chars), reference, autojunk=False)
    matched = 0
    for pdf_start, ref_start, size in matcher.get_matching_blocks():
        matched += size
        for offset in range(size):
            word_tokens[pdf_owner[pdf_start + offset]].add(
                reference_owner[ref_start + offset]
            )
    coverage = matched / max(1, len(reference))
    return tokens, word_tokens, coverage


def _pixel_box(pdf_box, page_height, image_width, image_height):
    left, bottom, right, top = pdf_box
    return (
        max(0, math.floor(left * RENDER_SCALE) - 1),
        max(0, math.floor((page_height - top) * RENDER_SCALE) - 1),
        min(image_width, math.ceil(right * RENDER_SCALE) + 1),
        min(image_height, math.ceil((page_height - bottom) * RENDER_SCALE) + 1),
    )


def _union(boxes):
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _annotation(annotation_id, image_id, box, text, **attributes):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 0,
        "bbox": [left, top, width, height],
        "area": width * height,
        "iscrowd": 0,
        "segmentation": [[left, top, right, top, right, bottom, left, bottom]],
        "attributes": {"text": text, **attributes},
    }


def _write_labels(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "text", "page", "order"])
        writer.writeheader()
        writer.writerows(rows)


def prepared(config):
    root = Path(config["download_dir"])
    version = root / ".preparation_version"
    required = [
        root / "pages",
        root / "word_detection.json",
        root / "line_detection.json",
        root / "word_recognition" / "labels.csv",
        root / "line_recognition" / "labels.csv",
        root / "metadata.json",
    ]
    return (
        version.is_file()
        and version.read_text(encoding="utf-8").strip() == str(PREPARATION_VERSION)
        and all(path.exists() for path in required)
    )


def prepare(config):
    if prepared(config):
        print(f"Skip preparation: already completed ({config['download_dir']})")
        return

    import pypdfium2 as pdfium

    root = Path(config["download_dir"])
    pdf_path = root / config["pdf_file"]
    text_path = root / config["text_file"]
    if not pdf_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(f"Missing source pair: {pdf_path}; {text_path}")

    temporary = root.with_name(f"{root.name}.prepare.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    pages_dir = temporary / "pages"
    word_dir = temporary / "word_recognition" / "images"
    line_dir = temporary / "line_recognition" / "images"
    for directory in (pages_dir, word_dir, line_dir):
        directory.mkdir(parents=True, exist_ok=True)

    corrected = text_path.read_text(encoding="utf-8-sig")
    references = re.split(r"\n\s*====page \d+====\s*\n", corrected)
    document = pdfium.PdfDocument(pdf_path)
    if len(document) != len(references):
        raise ValueError(
            f"PDF/TXT page mismatch: {len(document)} PDF pages, "
            f"{len(references)} text pages"
        )

    images = []
    word_annotations = []
    line_annotations = []
    word_rows = []
    line_rows = []
    coverages = []
    for page_index, (page, reference) in enumerate(zip(document, references)):
        bitmap = page.render(scale=RENDER_SCALE)
        image = bitmap.to_pil().convert("RGB")
        page_name = f"page_{page_index:03d}.jpg"
        image.save(pages_dir / page_name, quality=95)
        image_id = len(images)
        images.append(
            {"id": image_id, "file_name": page_name, "width": image.width, "height": image.height}
        )

        text_page = page.get_textpage()
        pdf_words = _pdf_words(text_page)
        tokens, assignments, coverage = _align_words(pdf_words, reference)
        coverages.append(coverage)
        page_height = page.get_height()
        token_boxes = {}
        token_lines = {}
        line_entries = {}
        line_boxes = {}
        for word_id, word in enumerate(pdf_words):
            box = _pixel_box(word["pdf_box"], page_height, image.width, image.height)
            line_boxes.setdefault(word["line"], []).append(box)
            for token_id in assignments[word_id]:
                token_boxes.setdefault(token_id, []).append(box)
                token_lines.setdefault(token_id, set()).add(word["line"])

        # Corrected tokens wholly contained by one visual line are safe for
        # word crops. A token hyphenated across two printed lines is excluded.
        for token_id, lines in token_lines.items():
            if len(lines) == 1:
                line_id = next(iter(lines))
                left = min(box[0] for box in token_boxes[token_id])
                line_entries.setdefault(line_id, []).append((left, tokens[token_id]))
            else:
                # Preserve visible fragments in line transcription when the
                # corrected TXT has joined a physical end-of-line hyphenation.
                for word_id, word in enumerate(pdf_words):
                    if token_id not in assignments[word_id]:
                        continue
                    fragment = word["text"].strip()
                    if fragment:
                        line_entries.setdefault(word["line"], []).append(
                            (word["pdf_box"][0] * RENDER_SCALE, fragment)
                        )

        for token_id in sorted(token_boxes):
            if len(token_lines[token_id]) != 1:
                continue
            box = _union(token_boxes[token_id])
            text = tokens[token_id]
            word_annotations.append(
                _annotation(len(word_annotations), image_id, box, text, order=token_id)
            )
            filename = f"{len(word_rows):07d}.jpg"
            image.crop(box).save(word_dir / filename, quality=95)
            word_rows.append({"image": filename, "text": text, "page": page_name, "order": token_id})

        for line_id in sorted(line_boxes):
            entries = sorted(line_entries.get(line_id, []))
            if not entries:
                continue
            box = _union(line_boxes[line_id])
            text = " ".join(value for _left, value in entries)
            line_annotations.append(
                _annotation(len(line_annotations), image_id, box, text, order=line_id)
            )
            filename = f"{len(line_rows):06d}.jpg"
            image.crop(box).save(line_dir / filename, quality=95)
            line_rows.append({"image": filename, "text": text, "page": page_name, "order": line_id})

    common = {"images": images, "categories": [{"id": 0, "name": "text"}]}
    (temporary / "word_detection.json").write_text(
        json.dumps({**common, "annotations": word_annotations}, ensure_ascii=False), encoding="utf-8"
    )
    (temporary / "line_detection.json").write_text(
        json.dumps({**common, "annotations": line_annotations}, ensure_ascii=False), encoding="utf-8"
    )
    _write_labels(temporary / "word_recognition" / "labels.csv", word_rows)
    _write_labels(temporary / "line_recognition" / "labels.csv", line_rows)
    metadata = {
        "dataset": "nevmenandr/russian-old-orthography-ocr",
        "book_id": BOOK_ID,
        "book_title": BOOK_TITLE,
        "pdf_pages": len(images),
        "word_samples": len(word_rows),
        "line_samples": len(line_rows),
        "mean_text_alignment_coverage": sum(coverages) / len(coverages),
        "geometry_source": "embedded PDF OCR text layer",
        "transcription_source": "paired human-readable TXT",
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Keep downloaded source files and replace only generated artifacts.
    for name in ("pages", "word_recognition", "line_recognition"):
        destination = root / name
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(temporary / name, destination)
    for name in ("word_detection.json", "line_detection.json", "metadata.json"):
        os.replace(temporary / name, root / name)
    shutil.rmtree(temporary)
    (root / ".preparation_version").write_text(str(PREPARATION_VERSION), encoding="utf-8")
    print(
        f"Prepared {BOOK_TITLE}: {len(images)} pages, {len(word_rows)} words, "
        f"{len(line_rows)} lines; alignment coverage {metadata['mean_text_alignment_coverage']:.2%}"
    )
