"""Benchmark the Total-Text MonkeyOCRv2 checkpoint as a word detector."""

from configs.detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.monkeyocrv2.run_benchmark import run


MODEL_NAME = "monkeyocrv2_dptext_totaltext_word"


if __name__ == "__main__":
    run(MODEL_NAME, BENCHMARKS[MODEL_NAME], DATASETS, "word")
