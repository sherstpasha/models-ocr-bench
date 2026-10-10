"""Benchmark the page-wise Party European-languages recognizer."""

import gc
import os
import urllib.request
import codecs
from statistics import fmean

import torch

from configs.recognition.benchmark_config import BENCHMARKS, DATASETS
from scripts.recognition import benchmark_kraken as common
from scripts.recognition.prepare_handwritten_essay import prepare_dataset


MODEL_NAMES = [
    name
    for name, config in BENCHMARKS.items()
    if config.get("backend") == "party" and config.get("run", False)
]


def ensure_model(config):
    path = config["model_path"]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(config["download_url"], partial)
        partial.replace(path)
    return path


def create_recognizer(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Party recognition benchmark requires CUDA")
    from lightning.fabric import Fabric
    from party.fusion import PartyModel
    from party.pred import batched_pred
    from party import tokenizer as party_tokenizer

    # Party's June 2025 tokenizer relies on the default ``length`` argument
    # added to int.to_bytes() in Python 3.11.  The isolated Party environment
    # intentionally uses Python 3.10, so keep the upstream algorithm but make
    # the one-byte conversion explicit.
    def decode_with_confs_py310(self, ids, confidences):
        lang_ids = {
            party_tokenizer.LANG_IDX_TO_ISO.get(
                int(token_id - party_tokenizer.LANG_OFFSET), "und"
            )
            for token_id in ids
            if token_id >= party_tokenizer.LANG_OFFSET
        }
        byte_ids = [
            token_id - party_tokenizer.OFFSET
            for token_id in ids
            if party_tokenizer.OFFSET
            <= token_id
            < party_tokenizer.LANG_OFFSET
        ]
        decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        chars, char_confs, pending_confs = [], [], []
        for byte_id, confidence in zip(bytes(byte_ids), confidences.tolist()):
            try:
                char = decoder.decode(bytes([byte_id]))
                pending_confs.append(confidence)
                if char:
                    chars.append(char)
                    char_confs.append(fmean(pending_confs))
                    pending_confs = []
            except UnicodeDecodeError:
                pending_confs = []
        return "".join(chars), char_confs, lang_ids

    party_tokenizer.OctetTokenizer.decode_with_confs = decode_with_confs_py310

    fabric = Fabric(accelerator="gpu", devices=[0], precision="bf16-mixed")
    with torch.inference_mode(), fabric.init_tensor(), fabric.init_module():
        model = PartyModel.from_safetensors(ensure_model(config))
    model = fabric.to_device(model)

    class Recognizer:
        def predict(self, image, segmentation):
            return batched_pred(
                model=model,
                im=image,
                bounds=segmentation,
                fabric=fabric,
                prompt_mode="curves",
                batch_size=config["batch_size"],
                add_lang_token=False,
            )

    return Recognizer()


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
