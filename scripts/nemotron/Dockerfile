FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG TARGETARCH
ARG TORCH_CUDA_ARCH_LIST=8.9
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip hatchling "setuptools>=68" psutil --root-user-action ignore

COPY models/nemotron-ocr-v2/nemotron-ocr /workspace/nemotron-ocr
WORKDIR /workspace/nemotron-ocr
RUN rm -f src/nemotron_ocr_cpp/*.so || true \
    && rm -rf build/ dist/
RUN --mount=type=cache,target=/root/.cache/pip \
    BUILD_CPP_FORCE=1 ARCH=${TARGETARCH} pip install -v . --no-build-isolation --root-user-action ignore

# Cache the backbone requested by the detector constructor so benchmark runs
# remain offline after the image has been built.
RUN python -c "from torch.hub import load_state_dict_from_url; load_state_dict_from_url('https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth', progress=False)"

WORKDIR /workspace/bench
