"""Benchmark ORLI's historical-document baseline detector."""

from pathlib import Path

from PIL import Image

from configs.line_detection.benchmark_config import BENCHMARKS
from scripts.line_detection.benchmark_external_common import run


MODEL_NAME = "orli_base"
CONFIG = BENCHMARKS[MODEL_NAME]


def ensure_model(config):
    model_path = Path(config["model_path"])
    if (
        model_path.is_file()
        and model_path.stat().st_size == config["expected_size"]
    ):
        # Loading below validates the completed safetensors file itself.
        return model_path
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    model_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(
            total=8,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )),
    )
    # Zenodo sometimes closes this large response mid-stream. Re-open it with
    # a Range request until the exact advertised size has been persisted.
    for stream_attempt in range(20):
        offset = model_path.stat().st_size if model_path.is_file() else 0
        if offset == config["expected_size"]:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(
                config["download_url"], headers=headers, stream=True, timeout=120
            ) as response:
                response.raise_for_status()
                resumed = offset > 0 and response.status_code == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    offset = 0
                written = offset
                next_report = written + 32 * 1024**2
                with model_path.open(mode) as output:
                    for chunk in response.iter_content(chunk_size=1024**2):
                        if not chunk:
                            continue
                        output.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            print(
                                f"ORLI checkpoint: {written / 1024**2:.0f} / "
                                f"{config['expected_size'] / 1024**2:.0f} MiB",
                                flush=True,
                            )
                            next_report += 32 * 1024**2
        except requests.RequestException as error:
            print(
                f"ORLI stream interrupted ({type(error).__name__}); "
                f"resume attempt {stream_attempt + 2}/20.",
                flush=True,
            )
            continue
    if model_path.stat().st_size != config["expected_size"]:
        raise IOError(
            f"Incomplete ORLI checkpoint: {model_path.stat().st_size} != "
            f"{config['expected_size']}"
        )
    return model_path


def create_model(config):
    # Importing ORLI registers its Kraken model and configuration plugins.
    from orli.configs import OrliSegmentationInferenceConfig
    from orli.orli import OrliModel  # noqa: F401
    from kraken.models import load_models

    models = load_models(ensure_model(config), tasks=["segmentation"])
    if len(models) != 1:
        raise RuntimeError(f"Expected one ORLI segmentation model, got {len(models)}")
    inference = OrliSegmentationInferenceConfig(
        accelerator="gpu",
        device=config["device"],
        precision=config["precision"],
        batch_size=1,
        polygonize=True,
        max_predicted_lines=config["max_predicted_lines"],
    )
    model = models[0]
    model.prepare_for_inference(inference)
    return model


def detect(model, image_path, _config):
    with Image.open(image_path) as image:
        result = model.predict(image.convert("RGB"))
    polygons = []
    for line in result.lines:
        if line.boundary and len(line.boundary) >= 3:
            polygons.append([[[float(x), float(y)] for x, y in line.boundary]])
    return polygons


if __name__ == "__main__":
    run(MODEL_NAME, CONFIG, create_model, detect, "ORLI Base")
