"""Everything that normally needs editing before an EAST benchmark run."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARKS = {
    "east_50_g1": {
        "run": True,
        "script": PROJECT_ROOT / "scripts" / "benchmark_east_50_g1.py",
        "output_dir": PROJECT_ROOT / "benchmark_results" / "east_50_g1",
        "weights": "east_50_g1",
        "preset": None,
        "target_size": 1408,
        "score_thresh": 0.6,
        "warmup_runs": 3,
        "cpu_only": False,
        "gpu_only": False,
    },
}

DATASETS = {
    "Archives020525": {
        "folder": Path(r"C:\shared\data0205\data02065\Archives020525\test_images"),
        "annotations": Path(r"C:\shared\data0205\data02065\Archives020525\test.json"),
    },
}
