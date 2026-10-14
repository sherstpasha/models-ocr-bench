FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV TORCH_CUDA_ARCH_LIST=8.9 \
    FORCE_CUDA=1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --root-user-action ignore \
      "numpy==2.4.4" "opencv-python-headless==4.13.0.92" "pillow==12.1.1" \
      "scipy==1.17.1" "timm==1.0.26" "shapely==2.1.2" \
      "albumentations==2.0.8" "Polygon3==3.0.9.1" "rapidfuzz==3.14.5" \
      "setuptools==80.10.2" "transformers==4.57.1" "safetensors==0.7.0" \
      "scikit-image==0.26.0" "editdistance==0.8.1" && \
    pip install --root-user-action ignore --no-build-isolation \
      "git+https://github.com/facebookresearch/detectron2.git@b599f139756bd3646a26a909caf86a1a159e53a7"

RUN git clone https://github.com/ymy-k/DPText-DETR.git /opt/dptext && \
    git -C /opt/dptext checkout c62baaed9da69b58280b2756720eb37c19a91058 && \
    git clone --depth 1 https://github.com/Yuliang-Liu/MonkeyOCRv2.git /tmp/monkeyocrv2 && \
    cp -r /tmp/monkeyocrv2/detection/DPText-DETR/configs/. /opt/dptext/configs/ && \
    cp -r /tmp/monkeyocrv2/detection/DPText-DETR/patch/adet/. /opt/dptext/adet/ && \
    cp -r /tmp/monkeyocrv2/detection/DPText-DETR/tools/. /opt/dptext/tools/ && \
    rm -rf /tmp/monkeyocrv2

WORKDIR /opt/dptext
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --root-user-action ignore --no-build-isolation --no-deps -e .

WORKDIR /workspace/bench
