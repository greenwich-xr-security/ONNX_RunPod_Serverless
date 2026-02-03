FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir runpod onnxruntime-gpu numpy Pillow

WORKDIR /app
COPY v2_m_age_regressor_ddp.onnx /app/v2_m_age_regressor_ddp.onnx
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
