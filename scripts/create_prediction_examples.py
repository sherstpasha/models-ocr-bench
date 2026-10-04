"""Build all compact prediction examples and embed them into README."""

from scripts.build_readme import main as build_readme
from scripts.create_detection_comparisons import render as render_detection
from scripts.recognition.create_prediction_collages import render as render_recognition


def main() -> None:
    for level in ("word_detection", "line_detection"):
        render_detection(level, seed=42)
    for level in ("word", "line"):
        render_recognition(level, examples_per_dataset=1, seed=42)
    build_readme()


if __name__ == "__main__":
    main()
