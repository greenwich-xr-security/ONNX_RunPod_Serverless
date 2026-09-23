# RunPod Serverless (MediaPipe + MobileSAM + Age Regressor)

Run a GPU-backed pipeline that:
1) detects hand joints with MediaPipe HandLandmarker,
2) selects the largest valid detected hand,
3) feeds that hand's joint points into MobileSAM's SAM predictor (vit_t, TinyViT backbone),
4) masks the input image to the hand region,
5) centres a square crop on the hand-joint bounding box, and
6) runs an ONNX age regressor.

When landmark-enabled inference cannot identify a valid set of 21 hand joints, it skips masking and ONNX inference and returns the existing response structure with `"age": -1.0` and `"std": 0.0`. No additional validity fields are added.

The worker supports:
- inference (`action=infer`, default),
- model catalog listing from GitHub Releases (`action=list_models`),
- model detail lookup (`action=get_model`),
- inference log storage/listing (`action=list_inference_logs`, `action=get_inference_log`).

Release sync runs on demand during model-catalog requests, not at worker startup.

## Files
- `handler.py`: RunPod serverless handler + local HTTP server
- `Dockerfile`: container build

## Model Sources
- Static mode (legacy): use `AGE_MODEL_PATH` / `AGE_MODEL_URL`
- Release-sync mode (new): sync all selected assets from GitHub Releases and serve a model catalog

In release-sync mode, source code archives are not downloaded because the worker only processes `assets[]` from the Releases API.

## Local Test (Docker)

Build:
```bash
docker build -t age-inference-runpod .
```

Build (legacy pre-bake models into image):
```bash
docker build -t age-inference-runpod --build-arg DOWNLOAD_MODELS=1 .
```

Run with release-sync mode + S3 inference logs:
```bash
docker run --gpus all -p 8000:8000 \
  -e MODEL_REPO_OWNER=<OWNER> \
  -e MODEL_REPO_NAME=<REPO> \
  -e GITHUB_TOKEN=<TOKEN_IF_PRIVATE> \
  -e SAVE_INFERENCE_LOGS=1 \
  -e INFERENCE_LOG_STORAGE=s3 \
  -e INFERENCE_LOG_S3_BUCKET=<S3_BUCKET> \
  -e INFERENCE_LOG_S3_ENDPOINT_URL=<S3_ENDPOINT_URL> \
  -e INFERENCE_LOG_S3_REGION=<S3_REGION> \
  -e INFERENCE_LOG_S3_ACCESS_KEY_ID=<S3_ACCESS_KEY_ID> \
  -e INFERENCE_LOG_S3_SECRET_ACCESS_KEY=<S3_SECRET_ACCESS_KEY> \
  age-inference-runpod
```

Windows PowerShell local workflow:
```powershell
. .\.vscode\set-env.local.ps1
docker run --gpus all -p 8000:8000 `
  -e MODEL_REPO_OWNER=$env:MODEL_REPO_OWNER `
  -e MODEL_REPO_NAME=$env:MODEL_REPO_NAME `
  -e GITHUB_TOKEN=$env:GITHUB_TOKEN `
  -e SAVE_INFERENCE_LOGS=1 `
  -e INFERENCE_LOG_STORAGE=s3 `
  -e INFERENCE_LOG_S3_BUCKET=$env:INFERENCE_LOG_S3_BUCKET `
  -e INFERENCE_LOG_S3_ENDPOINT_URL=$env:INFERENCE_LOG_S3_ENDPOINT_URL `
  -e INFERENCE_LOG_S3_REGION=$env:INFERENCE_LOG_S3_REGION `
  -e INFERENCE_LOG_S3_ACCESS_KEY_ID=$env:INFERENCE_LOG_S3_ACCESS_KEY_ID `
  -e INFERENCE_LOG_S3_SECRET_ACCESS_KEY=$env:INFERENCE_LOG_S3_SECRET_ACCESS_KEY `
  age-inference-runpod
```

## Local HTTP API

Health:
```bash
curl -s http://localhost:8000/health
```

List models:
```bash
curl -s http://localhost:8000/models
```

Get one model release:
```bash
curl -s http://localhost:8000/models/v2_m
```

Infer (latest/default model):
```bash
curl -s -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>"}}'
```

Infer with specific release tag:
```bash
curl -s -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>","model_tag":"v2_m"}}'
```

Infer with preprocessing flags:
```bash
curl -s -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>","model_tag":"v2_m","use_hand_landmarks":true,"use_hand_masking":true}}'
```

With `use_hand_landmarks=true` (the default), the selected hand is centred using a square around its landmark bounding box. The default crop reserves a 5% border on every side, so the longest landmark span occupies at most 90% of the final crop. If more than one valid hand is detected, the hand with the largest landmark bounding-box area is used. Setting both `use_hand_landmarks=false` and `use_hand_masking=false` retains the legacy whole-frame centre crop.

No-hand response (same response contract):
```json
{
  "age": -1.0,
  "std": 0.0,
  "model_tag": "v2_m",
  "model_name": "<MODEL_NAME>",
  "inference_id": null,
  "use_hand_landmarks": true,
  "use_hand_masking": true
}
```

List saved inference logs:
```bash
curl -s http://localhost:8000/inference-logs
```

Get one saved inference log:
```bash
curl -s http://localhost:8000/inference-logs/<INFERENCE_ID>
```

Get saved input image:
```bash
curl -s http://localhost:8000/inference-logs/<INFERENCE_ID>/image --output input.jpg
```

## RunPod API Usage

List models:
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"action":"list_models"}}'
```

Get release detail:
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"action":"get_model","model_tag":"v2_m"}}'
```

Infer using release tag:
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>","model_tag":"v2_m"}}'
```

Infer and control preprocessing:
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"image_base64":"<BASE64_OR_DATA_URL>","model_tag":"v2_m","use_hand_landmarks":true,"use_hand_masking":false}}'
```

List saved inference logs:
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"action":"list_inference_logs","limit":100}}'
```

Get one saved inference log (include input image as base64):
```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -d '{"input":{"action":"get_inference_log","inference_id":"<ID>","include_image":true}}'
```

## Configuration

Core:
- `FORCE_RUNPOD_SERVERLESS=1`: Force RunPod serverless mode locally.
- `MODEL_DOWNLOAD_TIMEOUT`: Download timeout seconds (default `1200`).

Release sync:
- `MODEL_REPO_OWNER`: GitHub owner/org.
- `MODEL_REPO_NAME`: GitHub repo name.
- `MODEL_REPO`: alternative `owner/repo` string.
- `GITHUB_TOKEN`: token for private repos/releases.
- `MODEL_STORE_DIR`: local cache directory (default `/app/models`).
- `MODEL_ASSET_PATTERNS`: comma-separated glob list (default `*.onnx,*.sha256,model_card.md,deployment_sheet.json`).
- `MODEL_MAX_RELEASES`: keep only first N releases from API order (0 = all).
- `MODEL_ALLOW_PRERELEASE`: include prereleases (`1`/`0`, default `1`).
- `MODEL_ALLOW_DRAFT`: include draft releases (`1`/`0`, default `0`).
- `MODEL_SYNC_MIN_INTERVAL_SEC`: minimum interval between release checks on model-catalog requests (default `300`).

Inference model fallback (legacy):
- `AGE_MODEL_PATH`: local model path (default `/app/v2_m_age_regressor_ddp.onnx`).
- `AGE_MODEL_URL`: fallback download URL.
- `AGE_MODEL_SHA256`: expected SHA256 for fallback download.

Segmentation stack:
- `HAND_LANDMARKER_PATH`: path to `hand_landmarker.task`.
- `HAND_LANDMARKER_URL`: download URL for hand landmarker.
- `MOBILE_SAM_CHECKPOINT`: path to `mobile_sam.pt`.
- `MOBILE_SAM_URL`: download URL for MobileSAM checkpoint.
- `MAX_HANDS`: maximum number of hands to detect (default `2`).
- `MIN_HAND_DET_CONF`: minimum detection confidence (default `0.5`).
- `MIN_HAND_PRESENCE_CONF`: minimum presence confidence (default `0.5`).
- `MIN_HAND_TRACKING_CONF`: minimum tracking confidence (default `0.5`).
- `HAND_CROP_BORDER_RATIO`: empty border reserved on each side of the centred hand crop (default `0.05`; must be at least `0.0` and less than `0.5`).
- `SAVE_MASKED_IMAGE=1`: save masked image (overwrites each request).
- `SAVE_MASKED_PATH`: path for masked image.

Inference logs:
- `SAVE_INFERENCE_LOGS=1`: enable saving inference input/output pairs.
- `INFERENCE_LOG_STORAGE`: must be `s3` (default `s3`).
- `INFERENCE_LOG_JPEG_QUALITY`: JPEG quality for saved inputs (default `90`).
- `INFERENCE_LOG_S3_BUCKET`: S3 bucket name for logs (required for S3 mode).
- `INFERENCE_LOG_S3_PREFIX`: key prefix inside bucket (default `inference_logs`).
- `INFERENCE_LOG_S3_ENDPOINT_URL`: S3-compatible endpoint URL (for RunPod volume S3 API).
- `INFERENCE_LOG_S3_REGION`: region for S3 client signing (default `us-east-1`).
- `INFERENCE_LOG_S3_ACCESS_KEY_ID`: optional explicit access key.
- `INFERENCE_LOG_S3_SECRET_ACCESS_KEY`: optional explicit secret key.

The worker writes:
- `<prefix>/index.json`
- `<prefix>/<INFERENCE_ID>.json`
- `<prefix>/<INFERENCE_ID>.jpg`
