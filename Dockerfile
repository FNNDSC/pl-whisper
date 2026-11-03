# pl-whisper: Python, CUDA 12's cuBLAS and cuDNN 9 from pip, Whisper large-v3, and this package.
#
# faster-whisper runs on CTranslate2, whose GPU path needs cuBLAS for CUDA 12 and
# cuDNN 9 and nothing else from CUDA; NVIDIA's pip wheels carry exactly those, so the
# base is plain Python rather than a CUDA image, and LD_LIBRARY_PATH names the wheels'
# library directories. The node's driver comes in through the NVIDIA container
# runtime; CUDA 12 runs on any driver from 525 on.
#
# The model is downloaded at build time, at a fixed revision, into /opt/whisper, and
# WHISPER_MODEL names it, so the plugin never needs the network; HF_HUB_OFFLINE makes
# sure of it. The model layer comes before the package so a code change does not
# download three gigabytes again.
#
# Build and check locally:
#
#   docker build -t localhost/fnndsc/pl-whisper:1.0.0 .
#   docker run --rm localhost/fnndsc/pl-whisper:1.0.0 pl-whisper --version
#   docker run --rm --gpus 1 -v $PWD/in:/incoming:ro -v $PWD/out:/outgoing \
#       localhost/fnndsc/pl-whisper:1.0.0 pl-whisper /incoming /outgoing

FROM docker.io/python:3.12-slim-trixie

LABEL org.opencontainers.image.authors="FNNDSC <dev@babyMRI.org>" \
      org.opencontainers.image.title="pl-whisper" \
      org.opencontainers.image.description="Whisper large-v3, through faster-whisper, over every recording of an input tree: a ChRIS DS plugin on pl-shexec" \
      org.opencontainers.image.source="https://github.com/FNNDSC/pl-whisper" \
      org.opencontainers.image.version="1.0.0" \
      whisper.model="Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478"

ARG SITE=/usr/local/lib/python3.12/site-packages
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    LD_LIBRARY_PATH=${SITE}/nvidia/cublas/lib:${SITE}/nvidia/cudnn/lib \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install --no-cache-dir "nvidia-cublas-cu12==12.*" "nvidia-cudnn-cu12==9.*"

ENV WHISPER_MODEL=/opt/whisper/large-v3
RUN pip install --no-cache-dir huggingface_hub \
    && python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3', revision='edaa852ec7e145841d8ffdb056a99866b5f0a478', local_dir='${WHISPER_MODEL}', allow_patterns=['config.json', 'model.bin', 'preprocessor_config.json', 'tokenizer.json', 'vocabulary.json'])" \
    && rm -rf ${WHISPER_MODEL}/.cache
ENV HF_HUB_OFFLINE=1

WORKDIR /usr/local/src/pl-whisper
COPY . .
RUN pip install --no-cache-dir .

CMD ["pl-whisper", "--man"]
