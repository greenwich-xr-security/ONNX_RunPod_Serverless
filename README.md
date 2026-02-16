# RunPod Serverless (MediaPipe + MobileSAM + Age Regressor)

Run a GPU-backed pipeline that:
1) detects hand joints with MediaPipe HandLandmarker,
2) feeds those joint points into MobileSAM's SAM predictor (vit_t, TinyViT backbone),
3) masks the input image to the hand region, and
4) runs the ONNX EfficientNet V2 M age regressor.

The worker accepts a base64 image and returns only `age` and `std` (same API as before).

## Files
- `handler.py`: RunPod serverless handler
- `Dockerfile`: container build

## Models
- `mobile_sam.pt` (MobileSAM vit_t checkpoint)
- `hand_landmarker.task` (MediaPipe HandLandmarker task)
- `v2_m_age_regressor_ddp.onnx` (EfficientNet V2 M age regressor)

By default, models are downloaded **on first run** if missing. To bake them into the image at build time, set `DOWNLOAD_MODELS=1`.

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

Notes:
- If no hands are detected or no mask is produced, the original image is used for age regression.

## Local Test (Docker)

Build:
```bash
docker build -t hand-segmentation-runpod .
```

Build (download models into the image):
```bash
docker build -t hand-segmentation-runpod --build-arg DOWNLOAD_MODELS=1 .
```

Run (GPU, local HTTP server on `:8000`):
```bash
docker run --gpus all -p 8000:8000 hand-segmentation-runpod
```

Run (CPU fallback, local HTTP server on `:8000`):
```bash
docker run -p 8000:8000 hand-segmentation-runpod
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

## RunPod Serverless

1. Create a RunPod Serverless endpoint.
2. Choose GitHub repository build.
3. Set `Dockerfile` path to `Dockerfile` and build context to `.`.
4. Build the image.
5. Deploy the endpoint.

### RunPod API Test (External)

Create an API key in RunPod Console -> Settings -> API Keys.

Async (queue-based):
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>"}}'
```

Check status:
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<REQUEST_ID> \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"
```

Sync (single request):
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>"}}'
```

## Configuration
- `HAND_LANDMARKER_PATH`: Path to `hand_landmarker.task` (default `/app/weights/hand_landmarker.task`).
- `MOBILE_SAM_CHECKPOINT`: Path to `mobile_sam.pt` (default `/app/weights/mobile_sam.pt`).
- `MAX_HANDS`: Maximum number of hands to detect (default `2`).
- `MIN_HAND_DET_CONF`: Minimum detection confidence (default `0.5`).
- `MIN_HAND_PRESENCE_CONF`: Minimum presence confidence (default `0.5`).
- `MIN_HAND_TRACKING_CONF`: Minimum tracking confidence (default `0.5`).
- `SAVE_MASKED_IMAGE=1`: Save the masked RGB image for inspection (overwrites on each request).
- `SAVE_MASKED_PATH`: Output path for the masked image (default `/app/masked_latest.png`).
- `FORCE_RUNPOD_SERVERLESS=1`: Force RunPod serverless mode locally.
