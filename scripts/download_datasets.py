from configs.detection.benchmark_config import DATASETS as DETECTION_DATASETS
from configs.line_detection.benchmark_config import DATASETS as LINE_DETECTION_DATASETS
from configs.recognition.benchmark_config import LINE_DATASETS, WORD_DATASETS
from utils.dataset_downloads import download_dataset_files


def missing_sources(config):
    if config.get("prepare") == "old_orthography" or config.get("format") == "old_orthography":
        required = [
            config["download_dir"] / config["pdf_file"],
            config["download_dir"] / config["text_file"],
        ]
    elif config.get("prepare") == "school_notebooks":
        required = [config["source_images"], config["source_annotations"]]
    elif "source_images" in config and "source_annotations" in config:
        required = [config["source_images"], config["source_annotations"]]
    elif "source_labels" in config:
        required = [config["folder"], config["source_labels"]]
    elif "source_annotations" in config:
        required = [config["folder"], config["source_annotations"]]
    elif "page_annotations" in config:
        required = [config["folder"], config["page_annotations"]]
    else:
        required = [config["folder"], config["annotations"]]
    return [path for path in required if not path.exists()]


def download_group(title, datasets):
    manual_steps = []
    print(f"\n## {title}")
    for name, config in datasets.items():
        print(f"\n### DATASET: {name}", flush=True)
        if config.get("manual"):
            message = (
                f"Manual dataset. Expected images: {config['folder']}; "
                f"annotations: {config['annotations']}"
            )
            manual_steps.append(message)
            print(message)
            continue
        try:
            download_dataset_files(name, config)
            if config.get("prepare") == "page_xml_lines":
                from scripts.line_detection.prepare_datasets import extract_huggingface_archives
                extract_huggingface_archives(config)
        except Exception as error:
            manual_steps.append(str(error))
            print(error)
        else:
            missing = missing_sources(config)
            if missing:
                message = "Extract or prepare manually. Expected: " + "; ".join(
                    str(path) for path in missing
                )
                manual_steps.append(message)
                print(message)
            else:
                print("Downloaded and source files found.")
    return manual_steps


def main():
    manual_steps = download_group("DETECTION", DETECTION_DATASETS)
    manual_steps.extend(download_group("LINE DETECTION", LINE_DETECTION_DATASETS))
    manual_steps.extend(download_group("WORD RECOGNITION", WORD_DATASETS))
    manual_steps.extend(download_group("LINE RECOGNITION", LINE_DATASETS))
    if manual_steps:
        print("\nDownloads completed. Manual steps:")
        for step in manual_steps:
            print(f"- {step}")
    else:
        print("\nAll datasets are available.")


if __name__ == "__main__":
    main()
