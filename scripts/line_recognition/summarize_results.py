"""Build the line-recognition summary from saved results."""

import os

os.environ["OCR_BENCH_RECOGNITION_LEVEL"] = "line"

from scripts.recognition.summarize_results import write_summary  # noqa: E402


if __name__ == "__main__":
    write_summary()
