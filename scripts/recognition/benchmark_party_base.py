"""Benchmark the current multilingual Llama Party base recognizer."""

import gc
import os

import torch

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
from scripts.recognition import benchmark_kraken as common
from scripts.recognition.benchmark_party import ensure_model
from scripts.recognition.prepare_handwritten_essay import prepare_dataset


MODEL_NAMES = [
    name
    for name, config in BENCHMARKS.items()
    if config.get("backend") == "party_base" and config.get("run", False)
]


def create_recognizer(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Party base benchmark requires CUDA")
    from kraken.models import load_models
    from party.configs import PartyRecognitionInferenceConfig

    models = load_models(ensure_model(config), tasks=["recognition"])
    if len(models) != 1:
        raise RuntimeError(f"Expected one recognition model, got {len(models)}")
    model = models[0]
    model.prepare_for_inference(PartyRecognitionInferenceConfig(
        accelerator="gpu",
        device=[0],
        batch_size=config["batch_size"],
        num_line_workers=0,
        precision="bf16-mixed",
        prompt_mode="curves",
        add_lang_token=False,
    ))
    return model


def main():
    args = common.parse_args()
    selected = os.environ.get("OCR_BENCH_ONLY_MODEL")
    for dataset_name, dataset_config in DATASETS.items():
        print(f"\n### DATASET: {dataset_name}")
        prepare_dataset(dataset_name, dataset_config)
    for model_name in MODEL_NAMES:
        if selected not in (None, model_name):
            continue
        config = BENCHMARKS[model_name]
        pending = []
        for dataset_name, dataset_config in DATASETS.items():
            result_path = config["output_dir"] / f"{dataset_name}_{model_name}.json"
            if args.force or not common.completed_result(result_path):
                pending.append((dataset_name, dataset_config))
        if not pending:
            continue
        gc.collect()
        torch.cuda.empty_cache()
        recognizer = create_recognizer(config)
        for dataset_name, dataset_config in pending:
            print(f"\n## MODEL: {model_name} / {dataset_name}")
            common.run_model(
                model_name,
                config,
                recognizer,
                dataset_name,
                dataset_config,
                force=args.force,
            )
        del recognizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
