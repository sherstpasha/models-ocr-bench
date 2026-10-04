$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

function Install-Environment {
    param([string]$Name, [string]$Requirements)
    if (-not (Test-Path "$Name\Scripts\python.exe")) {
        py -3.10 -m venv $Name
    }
    & "$Name\Scripts\python.exe" -m pip install --upgrade pip
    & "$Name\Scripts\python.exe" -m pip install -r $Requirements
}

Install-Environment ".venv" "requirements.txt"
& ".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime onnxruntime-gpu paddlepaddle paddlepaddle-gpu
& ".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt

Install-Environment ".venv-paddleocr" "requirements-paddleocr.txt"
Install-Environment ".venv-rfdetr" "requirements-rfdetr.txt"
Install-Environment ".venv-doc-ufcn" "requirements-doc-ufcn.txt"
Install-Environment ".venv-surya" "requirements-surya.txt"
Install-Environment ".venv-kraken" "requirements-kraken.txt"
Install-Environment ".venv-pero" "requirements-pero.txt"
Install-Environment ".venv-rtmdet" "requirements-rtmdet.txt"
& ".venv-rtmdet\Scripts\mim.exe" install "mmengine>=0.10" "mmcv>=2.1,<2.2" "mmdet>=3.3,<3.4"

if (-not (Get-Command tesseract -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id tesseract-ocr.tesseract --exact --accept-package-agreements --accept-source-agreements
    } else {
        Write-Warning "Tesseract 5 is not installed and winget is unavailable."
    }
}

& ".venv\Scripts\python.exe" -m scripts.download_models --group main
& ".venv-paddleocr\Scripts\python.exe" -m scripts.download_models --group paddleocr
& ".venv-rfdetr\Scripts\python.exe" -m scripts.download_models --group rfdetr
& ".venv-doc-ufcn\Scripts\python.exe" -m scripts.download_models --group doc-ufcn
& ".venv-surya\Scripts\python.exe" -m scripts.download_models --group surya
& ".venv-kraken\Scripts\python.exe" -m scripts.download_models --group kraken
& ".venv-pero\Scripts\python.exe" -m scripts.download_models --group pero
& ".venv-rtmdet\Scripts\python.exe" -m scripts.download_models --group rtmdet

Write-Host "All environments and models are ready."
