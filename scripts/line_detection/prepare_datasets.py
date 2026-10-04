"""Build line-level COCO annotations without copying source images."""

import json
import os
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from configs.line_detection.benchmark_config import DATASETS
from utils.dataset_downloads import download_dataset_files


TEXT_CATEGORIES = {"pupil_text", "pupil_comment", "teacher_comment"}
PREPARATION_VERSION = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def polygon_bbox(polygons):
    points = np.concatenate(
        [np.asarray(polygon, dtype=np.float64).reshape(-1, 2) for polygon in polygons]
    )
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def rectangle_segmentation(bbox):
    x, y, width, height = bbox
    return [[x, y, x + width, y, x + width, y + height, x, y + height]]


def polygon_area(polygon):
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    return float(
        abs(
            np.dot(points[:, 0], np.roll(points[:, 1], 1))
            - np.dot(points[:, 1], np.roll(points[:, 0], 1))
        )
        / 2
    )


def coco_polygons(polygons):
    return [np.asarray(polygon, dtype=np.float64).reshape(-1).tolist() for polygon in polygons]


def line_hull(polygons):
    """Return one convex line envelope spanning all words and their gaps."""
    points = np.concatenate(
        [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in polygons]
    )
    # Andrew's monotone-chain hull keeps this preparation independent of OpenCV.
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return [list(point) for point in unique]

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return [list(point) for point in lower[:-1] + upper[:-1]]


def point_in_polygon(point, polygon):
    x, y = point
    vertices = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def prepare_handwritten_essay(config):
    pages = json.loads(config["source_annotations"].read_text(encoding="utf-8"))
    source_coco = json.loads(config["source_coco"].read_text(encoding="utf-8"))
    dimensions = {
        image["file_name"]: (image["width"], image["height"])
        for image in source_coco["images"]
    }
    images = []
    annotations = []
    for page_name, page in pages.items():
        if page_name not in dimensions:
            raise KeyError(f"Missing image metadata for {page_name}")
        image_id = len(images)
        width, height = dimensions[page_name]
        images.append(
            {"id": image_id, "file_name": page_name, "width": width, "height": height}
        )
        lines = [
            line
            for block in page.get("blocks", [])
            for line in block.get("lines", [])
        ] + list(page.get("free_lines", []))
        lines.sort(key=lambda line: line.get("order", 0))
        for line_order, line in enumerate(lines):
            polygons = [word.get("polygon") for word in line.get("words", [])]
            polygons = [polygon for polygon in polygons if polygon]
            if not polygons:
                continue
            bbox = polygon_bbox(polygons)
            hull = line_hull(polygons)
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": 0,
                    "bbox": bbox,
                    "area": polygon_area(hull),
                    "iscrowd": 0,
                    "segmentation": coco_polygons([hull]),
                    "attributes": {"line_order": line_order},
                }
            )
    return images, annotations, {"source_lines": len(annotations)}


def prepare_school_notebooks(config):
    source = json.loads(config["source_annotations"].read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in source["categories"]}
    text_ids = {key for key, value in category_names.items() if value in TEXT_CATEGORIES}
    line_ids = {key for key, value in category_names.items() if value == "text_line"}
    paper_ids = {key for key, value in category_names.items() if value == "paper"}
    by_image = {}
    for annotation in source["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)

    output = []
    ungrouped_words = 0
    for image in source["images"]:
        items = by_image.get(image["id"], [])
        words = [item for item in items if item["category_id"] in text_ids]
        ungrouped_words += sum(item.get("group_id") is None for item in words)
        grouped = {}
        for word in words:
            if word.get("group_id") is not None and word.get("segmentation"):
                grouped.setdefault(word["group_id"], []).append(word)

        baselines = [item for item in items if item["category_id"] in line_ids]
        if len(grouped) != len(baselines):
            raise ValueError(
                f"{image['file_name']}: {len(grouped)} word groups but "
                f"{len(baselines)} text_line annotations"
            )

        papers = [item for item in items if item["category_id"] in paper_ids]
        papers.sort(key=lambda item: polygon_bbox(item["segmentation"])[0])
        records = []
        for group_id, group_words in grouped.items():
            polygons = [part for word in group_words for part in word["segmentation"]]
            bbox = polygon_bbox(polygons)
            hull = line_hull(polygons)
            center = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
            page_index = next(
                (
                    index
                    for index, paper in enumerate(papers)
                    if any(point_in_polygon(center, part) for part in paper["segmentation"])
                ),
                -1,
            )
            records.append(
                (page_index, bbox[1] + bbox[3] / 2, bbox[0], group_id, bbox, hull)
            )

        records.sort(key=lambda item: (item[0], item[1], item[2]))
        order_by_page = {}
        for page_index, _y, _x, group_id, bbox, hull in records:
            line_order = order_by_page.get(page_index, 0)
            order_by_page[page_index] = line_order + 1
            output.append(
                {
                    "id": len(output),
                    "image_id": image["id"],
                    "category_id": 0,
                    "bbox": bbox,
                    "area": polygon_area(hull),
                    "iscrowd": 0,
                    "segmentation": coco_polygons([hull]),
                    "attributes": {
                        "group_id": group_id,
                        "page_index": page_index,
                        "line_order": line_order,
                    },
                }
            )
    return source["images"], output, {
        "source_lines": sum(
            item["category_id"] in line_ids for item in source["annotations"]
        ),
        "ungrouped_words_excluded": ungrouped_words,
    }


def extract_huggingface_archives(config):
    """Extract this dataset's archives while preventing paths outside the target."""
    destination = Path(config["download_dir"])
    targets = {
        "data/images": Path(config.get("source_images", config["folder"])),
        "data/page_xmls": Path(config["source_annotations"]),
    }
    for prefix, target in targets.items():
        target.mkdir(parents=True, exist_ok=True)
        for filename in config["download_files"]:
            if not filename.startswith(prefix + "/"):
                continue
            marker = target / f".{Path(filename).name}.extracted"
            if marker.is_file():
                continue
            archive = destination / filename
            target_root = target.resolve()
            with tarfile.open(archive, "r:gz") as bundle:
                for member in bundle.getmembers():
                    member_path = (target / member.name).resolve()
                    if target_root != member_path and target_root not in member_path.parents:
                        raise RuntimeError(f"Unsafe archive member: {member.name}")
                bundle.extractall(target)
            marker.touch()
            print(f"Extracted {archive.name} into {target}")


def _local_name(element):
    return element.tag.rsplit("}", 1)[-1]


def _page_xml_points(element):
    coords = next((child for child in element if _local_name(child) == "Coords"), None)
    if coords is None:
        return []
    return [
        [float(x), float(y)]
        for point in coords.attrib.get("points", "").split()
        for x, y in [point.split(",", 1)]
    ]


def prepare_page_xml_lines(config):
    images_root = Path(config.get("source_images", config["folder"]))
    benchmark_root = Path(config["folder"])
    benchmark_root.mkdir(parents=True, exist_ok=True)
    xml_root = Path(config["source_annotations"])
    image_paths = [
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    by_relative_stem = {
        path.relative_to(images_root).with_suffix("").as_posix(): path
        for path in image_paths
    }
    by_stem = {}
    for path in image_paths:
        by_stem.setdefault(path.stem, []).append(path)

    xml_paths = sorted(
        xml_root.rglob("*.xml"),
        key=lambda path: path.relative_to(xml_root).as_posix(),
    )
    total_source_images = len(xml_paths)
    max_images = config.get("max_images")
    if max_images is not None:
        xml_paths = xml_paths[: int(max_images)]

    images = []
    annotations = []
    missing_images = []
    benchmark_paths = set()
    for xml_path in xml_paths:
        relative_stem = xml_path.relative_to(xml_root).with_suffix("").as_posix()
        image_path = by_relative_stem.get(relative_stem)
        if image_path is None:
            candidates = by_stem.get(xml_path.stem, [])
            image_path = candidates[0] if len(candidates) == 1 else None
        if image_path is None:
            missing_images.append(str(xml_path.relative_to(xml_root)))
            continue

        root = ET.parse(xml_path).getroot()
        page = next((item for item in root.iter() if _local_name(item) == "Page"), None)
        if page is None:
            raise ValueError(f"PAGE XML has no Page element: {xml_path}")
        image_id = len(images)
        benchmark_name = f"image_{image_id:05d}{image_path.suffix.lower()}"
        benchmark_path = benchmark_root / benchmark_name
        benchmark_paths.add(benchmark_path.resolve())
        if not benchmark_path.is_file():
            try:
                os.link(image_path, benchmark_path)
            except OSError:
                shutil.copy2(image_path, benchmark_path)
        images.append({
            "id": image_id,
            "file_name": benchmark_name,
            "width": int(page.attrib["imageWidth"]),
            "height": int(page.attrib["imageHeight"]),
        })
        line_order = 0
        for line in (item for item in page.iter() if _local_name(item) == "TextLine"):
            polygon = _page_xml_points(line)
            if len(polygon) < 3:
                continue
            bbox = polygon_bbox([polygon])
            annotations.append({
                "id": len(annotations),
                "image_id": image_id,
                "category_id": 0,
                "bbox": bbox,
                "area": polygon_area(polygon),
                "iscrowd": 0,
                "segmentation": coco_polygons([polygon]),
                "attributes": {"line_order": line_order},
            })
            line_order += 1
    if missing_images:
        examples = ", ".join(missing_images[:3])
        raise FileNotFoundError(
            f"No matching image for {len(missing_images)} PAGE XML files; examples: {examples}"
        )
    if not images:
        raise RuntimeError(f"No PAGE XML/image pairs found under {xml_root} and {images_root}")
    for stale_path in benchmark_root.glob("image_*.*"):
        if stale_path.resolve() not in benchmark_paths:
            stale_path.unlink()
    return images, annotations, {
        "source_lines": len(annotations),
        "total_source_images": total_source_images,
        "selected_images": len(images),
        "selection": "first_relative_page_xml_paths",
    }


def prepare_dataset(name, config, force=False):
    if config.get("prepare") == "old_orthography":
        from scripts.prepare_old_orthography import prepare

        download_dataset_files(name, config)
        prepare(config)
        return
    destination = Path(config["annotations"])
    if destination.is_file() and not force:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("preparation", {}).get("version") == PREPARATION_VERSION:
            print(f"Skip preparation: already completed ({destination})")
            return
        print(f"Rebuilding outdated line annotations: {destination}")
    if config["prepare"] == "page_xml_lines":
        download_dataset_files(name, config)
        extract_huggingface_archives(config)
    source_annotations = Path(config["source_annotations"])
    source_exists = (
        source_annotations.is_dir()
        if config["prepare"] == "page_xml_lines"
        else source_annotations.is_file()
    )
    if not source_exists:
        raise FileNotFoundError(f"Missing source_annotations: {source_annotations}")
    if config["prepare"] == "handwritten_essay_lines":
        if not Path(config["source_coco"]).is_file():
            raise FileNotFoundError(f"Missing source_coco: {config['source_coco']}")
        images, annotations, stats = prepare_handwritten_essay(config)
    elif config["prepare"] == "school_notebooks_lines":
        images, annotations, stats = prepare_school_notebooks(config)
    elif config["prepare"] == "page_xml_lines":
        images, annotations, stats = prepare_page_xml_lines(config)
    else:
        raise ValueError(f"Unknown preparation mode: {config['prepare']}")
    payload = {
        "categories": [{"id": 0, "name": "text_line"}],
        "images": images,
        "annotations": annotations,
        "preparation": {**stats, "version": PREPARATION_VERSION},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(annotations)} lines in {len(images)} images: {destination}")
    if stats.get("ungrouped_words_excluded"):
        print(f"Excluded ungrouped words: {stats['ungrouped_words_excluded']}")


def main():
    for name, config in DATASETS.items():
        print(f"\nPreparing {name}...")
        prepare_dataset(name, config)


if __name__ == "__main__":
    main()
