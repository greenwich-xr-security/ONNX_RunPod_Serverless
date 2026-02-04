FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir runpod onnxruntime-gpu numpy Pillow

WORKDIR /app
RUN curl -fsSL -o /app/v2_m_age_regressor_ddp.onnx \
        https://github.com/greenwich-xr-security/ONNX_RunPod_Serverless/releases/download/1/v2_m_age_regressor_ddp.onnx \
    && echo "618a3935d3e5a15c9f7ec3f39fe759b2239497d5184b146a8c55dc6660add395  /app/v2_m_age_regressor_ddp.onnx" | sha256sum -c -
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
