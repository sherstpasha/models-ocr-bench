#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

install_environment() {
    local name="$1"
    local requirements="$2"
    if [[ ! -x "$name/bin/python" ]]; then
        "$PYTHON_BIN" -m venv "$name"
    fi
    "$name/bin/python" -m pip install --upgrade pip
    "$name/bin/python" -m pip install -r "$requirements"
}

install_environment .venv requirements.txt
.venv/bin/python -m pip uninstall -y onnxruntime onnxruntime-gpu paddlepaddle paddlepaddle-gpu || true
.venv/bin/python -m pip install -r requirements-gpu.txt

install_environment .venv-paddleocr requirements-paddleocr.txt
install_environment .venv-rfdetr requirements-rfdetr.txt
install_environment .venv-doc-ufcn requirements-doc-ufcn.txt
install_environment .venv-surya requirements-surya.txt
install_environment .venv-kraken requirements-kraken.txt
install_environment .venv-pero requirements-pero.txt
install_environment .venv-rtmdet requirements-rtmdet.txt
.venv-rtmdet/bin/mim install 'mmengine>=0.10' 'mmcv>=2.1,<2.2' 'mmdet>=3.3,<3.4'

if ! command -v tesseract >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
    else
        echo 'WARNING: install Tesseract 5 manually.' >&2
    fi
fi

.venv/bin/python -m scripts.download_models --group main
.venv-paddleocr/bin/python -m scripts.download_models --group paddleocr
.venv-rfdetr/bin/python -m scripts.download_models --group rfdetr
.venv-doc-ufcn/bin/python -m scripts.download_models --group doc-ufcn
.venv-surya/bin/python -m scripts.download_models --group surya
.venv-kraken/bin/python -m scripts.download_models --group kraken
.venv-pero/bin/python -m scripts.download_models --group pero
.venv-rtmdet/bin/python -m scripts.download_models --group rtmdet

echo 'All environments and models are ready.'
