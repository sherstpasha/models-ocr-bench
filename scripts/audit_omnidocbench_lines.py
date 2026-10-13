"""Build a visual audit of OmniDocBench pages suitable for line detection."""

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont


REPOSITORY = "HKolt/OmniDocBench"
TEXT_BLOCK = "text_block"


def has_line_spans(element):
    if element.get("line_with_spans"):
        return True
    return any(has_line_spans(child) for child in element.get("merge_list", []))


def collect_spans(element, output):
    output.extend(element.get("line_with_spans", []))
    for child in element.get("merge_list", []):
        collect_spans(child, output)


def coordinates(values):
    return [(values[index], values[index + 1]) for index in range(0, len(values), 2)]


def is_candidate(page):
    text_blocks = [
        element
        for element in page["layout_dets"]
        if element.get("category_type") == TEXT_BLOCK and element.get("text")
    ]
    spans = []
    for element in page["layout_dets"]:
        collect_spans(element, spans)
    text_spans = [span for span in spans if span.get("category_type") == "text_span"]
    return bool(text_spans) and bool(text_blocks) and all(
        has_line_spans(element) for element in text_blocks
    )


def load_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(page, image_path, output_path, maximum_height):
    source = Image.open(image_path).convert("RGB")
    overlay = source.convert("RGBA")
    drawing_layer = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(drawing_layer)
    line_width = max(3, round(max(source.size) / 800))
    spans = []
    ignore_regions = []

    for element in page["layout_dets"]:
        collect_spans(element, spans)
        polygon = element.get("poly") or []
        if len(polygon) >= 8 and not has_line_spans(element):
            points = coordinates(polygon)
            draw.polygon(points, fill=(220, 65, 65, 32))
            draw.line(points + [points[0]], fill=(190, 35, 35, 235), width=line_width)
            ignore_regions.append(element)

    for span in spans:
        polygon = span.get("poly") or []
        if len(polygon) < 8:
            continue
        points = coordinates(polygon)
        if span.get("category_type") == "text_span":
            outline, fill = (20, 145, 65, 245), (35, 190, 80, 32)
        else:
            outline, fill = (225, 145, 15, 245), (240, 165, 25, 40)
        draw.polygon(points, fill=fill)
        draw.line(points + [points[0]], fill=outline, width=line_width)

    overlay = Image.alpha_composite(overlay, drawing_layer).convert("RGB")
    scale = min(1.0, maximum_height / source.height)
    size = (round(source.width * scale), round(source.height * scale))
    source = source.resize(size, Image.Resampling.LANCZOS)
    overlay = overlay.resize(size, Image.Resampling.LANCZOS)
    header = 88
    canvas = Image.new("RGB", (size[0] * 2, size[1] + header), "white")
    canvas.paste(source, (0, header))
    canvas.paste(overlay, (size[0], header))
    painter = ImageDraw.Draw(canvas)
    painter.text((12, 8), "Original", fill=(25, 25, 25), font=load_font(26))
    painter.text((size[0] + 12, 8), "Annotation audit", fill=(25, 25, 25), font=load_font(26))
    painter.text(
        (size[0] + 12, 48),
        "GREEN line GT   RED ignore candidate   ORANGE non-text span",
        fill=(25, 25, 25),
        font=load_font(19),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90, optimize=True)
    return spans, ignore_regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/dataset_audits/omnidocbench_line_candidates"),
    )
    parser.add_argument("--maximum-height", type=int, default=1100)
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "source"
    previews = output / "previews"
    annotation_path = Path(
        hf_hub_download(
            REPOSITORY,
            "OmniDocBench.json",
            repo_type="dataset",
            local_dir=source,
        )
    )
    pages = json.loads(annotation_path.read_text(encoding="utf-8"))
    candidates = [(index, page) for index, page in enumerate(pages) if is_candidate(page)]
    records = []

    for number, (source_index, page) in enumerate(candidates, start=1):
        info = page["page_info"]
        image_name = info["image_path"]
        image_path = Path(
            hf_hub_download(
                REPOSITORY,
                f"images/{image_name}",
                repo_type="dataset",
                local_dir=source,
            )
        )
        preview_name = f"{number:03d}_{Path(image_name).stem}.jpg"
        spans, ignore_regions = render(
            page, image_path, previews / preview_name, args.maximum_height
        )
        attributes = info.get("page_attribute", {})
        records.append(
            {
                "candidate_number": number,
                "source_index": source_index,
                "image_path": image_name,
                "preview": f"previews/{preview_name}",
                "language": attributes.get("language", ""),
                "data_source": attributes.get("data_source", ""),
                "layout": attributes.get("layout", ""),
                "subset": attributes.get("subset", ""),
                "text_lines": sum(
                    span.get("category_type") == "text_span" for span in spans
                ),
                "other_spans": sum(
                    span.get("category_type") != "text_span" for span in spans
                ),
                "ignore_candidates": len(ignore_regions),
                "review": "pending",
                "notes": "",
            }
        )
        print(f"[{number}/{len(candidates)}] {image_name}", flush=True)

    fieldnames = list(records[0]) if records else []
    with (output / "pages.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    (output / "pages.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    gallery = ["# OmniDocBench line candidates", ""]
    for record in records:
        gallery.extend(
            [
                f"## {record['candidate_number']:03d}: `{record['image_path']}`",
                "",
                f"Language: `{record['language']}` · lines: {record['text_lines']} · "
                f"ignore regions: {record['ignore_candidates']} · review: **pending**",
                "",
                f"![{record['image_path']}]({record['preview']})",
                "",
            ]
        )
    (output / "GALLERY.md").write_text("\n".join(gallery), encoding="utf-8")
    (output / "README.md").write_text(
        "# OmniDocBench line-detection candidate audit\n\n"
        f"Candidate pages: {len(records)}. Review `previews/` and set `review` in "
        "`pages.csv` to `keep` or `drop`; `GALLERY.md` displays every page.\n\n"
        "Selection requires at least one `text_span` and line spans for every "
        "top-level `text_block`.\n\n"
        "- Green: `text_span`, proposed line ground truth.\n"
        "- Red: page region without line spans, proposed ignore region.\n"
        "- Orange: inline equation, footnote mark, or another non-text span.\n",
        encoding="utf-8",
    )
    print(f"Audit written to {output}")


if __name__ == "__main__":
    main()
