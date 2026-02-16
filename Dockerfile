FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libglib2.0-0 \
        libgl1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir \
        mediapipe \
        "numpy<2" \
        onnxruntime-gpu \
        opencv-python-headless \
        Pillow \
        runpod \
        timm \
    && python3 -m pip install --no-cache-dir \
        torch==2.2.2 \
        torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu118 \
    && python3 -m pip install --no-cache-dir \
        git+https://github.com/ChaoningZhang/MobileSAM.git

WORKDIR /app
ARG DOWNLOAD_MODELS=0
RUN mkdir -p /app/weights \
    && if [ "$DOWNLOAD_MODELS" = "1" ]; then \
        curl -fL -o /app/v2_m_age_regressor_ddp.onnx \
            https://github.com/greenwich-xr-security/ONNX_RunPod_Serverless/releases/download/1/v2_m_age_regressor_ddp.onnx \
        && echo "618a3935d3e5a15c9f7ec3f39fe759b2239497d5184b146a8c55dc6660add395  /app/v2_m_age_regressor_ddp.onnx" | sha256sum -c - \
        && curl -fL -o /app/weights/mobile_sam.pt \
            https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt \
        && curl -fL -o /app/weights/hand_landmarker.task \
            https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task; \
    fi

COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
