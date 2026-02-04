# ONNX RunPod Serverless (Age Regressor)

Run a GPU-backed ONNX EfficientNet V2 M age regressor on RunPod Serverless. The worker accepts a base64 image and returns only `age` and `std`.

## Files
- `v2_m_age_regressor_ddp.onnx`: model (input size 480x480, EfficientNet V2 M)
- `handler.py`: RunPod serverless handler
- `Dockerfile`: container build

## Input / Output

Request body:
```json
{
  "input": {
    "image_base64": "<base64-encoded image or data URL>"
  }
}
```

Response body:
```json
{
  "age": 23.4,
  "std": 4.9
}
```

## Local Test (Docker)

Build:
```bash
docker build -t onnx-runpod .
```

Run (GPU, local HTTP server on `:8000`):
```bash
docker run --gpus all -p 8000:8000 onnx-runpod
```

Run (CPU fallback, local HTTP server on `:8000`):
```bash
docker run -p 8000:8000 onnx-runpod
```

Create base64 payload:
```bash
python - <<'PY'
import base64
from pathlib import Path
print(base64.b64encode(Path("test.jpg").read_bytes()).decode())
PY
```

Call the local worker:
```bash
curl -s -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"input":{"image_base64":"<PASTE_BASE64_HERE>"}}'
```

Expected output:
```json
{"age": 23.4, "std": 4.9}
```

## RunPod Serverless

1. Build and push the image to your registry.
2. Create a Serverless endpoint in RunPod using the image.
3. Send the same JSON payload as above.

## Notes
- Preprocessing uses center-crop to square, resize to model input size, and ImageNet mean/std normalization.
- Output uses `age = mean`, `std = exp(0.5 * log_var)`.
- The handler supports either two outputs (`mean`, `log_var`) or a single output tensor with at least two values.
- The container runs a local HTTP server by default. Set `FORCE_RUNPOD_SERVERLESS=1` to force RunPod serverless mode locally.
