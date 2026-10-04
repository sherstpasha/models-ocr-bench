from configs.recognition.benchmark_config import DATASETS
from utils.dataset_downloads import download_dataset_files


def main():
    manual_steps = []
    for name, config in DATASETS.items():
        print(f"\n### DATASET: {name}", flush=True)
        try:
            download_dataset_files(name, config)
        except FileNotFoundError as error:
            manual_steps.append(str(error))
            print(error)
        else:
            print("Downloaded and source files found.")

    if manual_steps:
        print("\nDownloads completed. Manual extraction is required:")
        for step in manual_steps:
            print(f"- {step}")
    else:
        print("\nAll recognition datasets are available.")


if __name__ == "__main__":
    main()
