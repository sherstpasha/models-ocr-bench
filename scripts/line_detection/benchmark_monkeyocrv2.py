"""Benchmark the CTW1500 MonkeyOCRv2 checkpoint as a line detector."""

from configs.line_detection.benchmark_config import BENCHMARKS, DATASETS
from scripts.monkeyocrv2.run_benchmark import run


MODEL_NAME = "monkeyocrv2_dptext_ctw1500_line"


if __name__ == "__main__":
    run(MODEL_NAME, BENCHMARKS[MODEL_NAME], DATASETS, "line")
