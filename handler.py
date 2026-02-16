import base64
import hashlib
import io
import json
import os
import shutil
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image
import runpod
import torch

HAND_LANDMARKER_PATH = os.environ.get(
    "HAND_LANDMARKER_PATH", "/app/weights/hand_landmarker.task"
)
HAND_LANDMARKER_URL = os.environ.get(
    "HAND_LANDMARKER_URL",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
)
MOBILE_SAM_CHECKPOINT = os.environ.get(
    "MOBILE_SAM_CHECKPOINT", "/app/weights/mobile_sam.pt"
)
MOBILE_SAM_URL = os.environ.get(
    "MOBILE_SAM_URL",
    "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt",
)
AGE_MODEL_PATH = os.environ.get(
    "AGE_MODEL_PATH", "/app/v2_m_age_regressor_ddp.onnx"
)
AGE_MODEL_URL = os.environ.get(
    "AGE_MODEL_URL",
    "https://github.com/greenwich-xr-security/ONNX_RunPod_Serverless/releases/download/1/v2_m_age_regressor_ddp.onnx",
)
AGE_MODEL_SHA256 = os.environ.get(
    "AGE_MODEL_SHA256",
    "618a3935d3e5a15c9f7ec3f39fe759b2239497d5184b146a8c55dc6660add395",
)
MODEL_DOWNLOAD_TIMEOUT = int(os.environ.get("MODEL_DOWNLOAD_TIMEOUT", "1200"))
SAVE_MASKED_IMAGE = os.environ.get("SAVE_MASKED_IMAGE", "0") == "1"
SAVE_MASKED_PATH = os.environ.get("SAVE_MASKED_PATH", "/app/masked_latest.png")
MAX_HANDS = int(os.environ.get("MAX_HANDS", "2"))
MIN_HAND_DET_CONF = float(os.environ.get("MIN_HAND_DET_CONF", "0.5"))
MIN_HAND_PRESENCE_CONF = float(os.environ.get("MIN_HAND_PRESENCE_CONF", "0.5"))
MIN_HAND_TRACKING_CONF = float(os.environ.get("MIN_HAND_TRACKING_CONF", "0.5"))

_HAND_LANDMARKER: vision.HandLandmarker | None = None
_SAM_PREDICTOR: SamPredictor | None = None
_DEVICE: str | None = None
_SESSION: ort.InferenceSession | None = None
_INPUT_NAME: str | None = None
_INPUT_SIZE: int | None = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INPUT_SIZE = 480


def _get_running_mode() -> Any:
    if hasattr(vision, "RunningMode"):
        return vision.RunningMode.IMAGE
    if hasattr(vision, "VisionRunningMode"):
        return vision.VisionRunningMode.IMAGE
    raise RuntimeError("Unsupported mediapipe vision running mode.")


def _ensure_file(path: str, url: str, sha256: str | None = None) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with urllib.request.urlopen(url, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
            with open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
        if sha256:
            hasher = hashlib.sha256()
            with open(tmp_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if digest.lower() != sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {path}: expected {sha256}, got {digest}"
                )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_hand_landmarker() -> vision.HandLandmarker:
    global _HAND_LANDMARKER
    if _HAND_LANDMARKER is None:
        if not os.path.exists(HAND_LANDMARKER_PATH):
            _ensure_file(HAND_LANDMARKER_PATH, HAND_LANDMARKER_URL)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
            running_mode=_get_running_mode(),
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=MIN_HAND_DET_CONF,
            min_hand_presence_confidence=MIN_HAND_PRESENCE_CONF,
            min_tracking_confidence=MIN_HAND_TRACKING_CONF,
        )
        _HAND_LANDMARKER = vision.HandLandmarker.create_from_options(options)
    return _HAND_LANDMARKER


def _load_sam_predictor() -> SamPredictor:
    global _SAM_PREDICTOR, _DEVICE
    if _SAM_PREDICTOR is None:
        if not os.path.exists(MOBILE_SAM_CHECKPOINT):
            _ensure_file(MOBILE_SAM_CHECKPOINT, MOBILE_SAM_URL)
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_t"](checkpoint=MOBILE_SAM_CHECKPOINT)
        sam.to(device=_DEVICE)
        sam.eval()
        _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR


def _infer_img_size(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        if shape[2] == shape[3]:
            return int(shape[2])
    return None


def _load_age_model() -> Tuple[ort.InferenceSession, str, int]:
    global _SESSION, _INPUT_NAME, _INPUT_SIZE
    if _SESSION is None:
        if not os.path.exists(AGE_MODEL_PATH):
            _ensure_file(AGE_MODEL_PATH, AGE_MODEL_URL, AGE_MODEL_SHA256)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _SESSION = ort.InferenceSession(AGE_MODEL_PATH, providers=providers)
        _INPUT_NAME = _SESSION.get_inputs()[0].name
        _INPUT_SIZE = _infer_img_size(_SESSION) or DEFAULT_INPUT_SIZE
    return _SESSION, _INPUT_NAME, _INPUT_SIZE


def _decode_image(job_input: Dict[str, Any]) -> Image.Image:
    if "image_base64" not in job_input:
        raise ValueError("Missing required field: image_base64")
    data = job_input["image_base64"]
    if isinstance(data, str) and data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ValueError("image_base64 is not valid base64 data") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _prepare_image(img: Image.Image, size: int) -> np.ndarray:
    width, height = img.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    img = img.crop((left, top, left + crop, top + crop)).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


def _extract_mean_logvar(outputs: list[np.ndarray]) -> Tuple[float, float]:
    if len(outputs) >= 2:
        mean = outputs[0]
        log_var = outputs[1]
    elif len(outputs) == 1:
        out = np.asarray(outputs[0])
        flat = out.reshape(-1, out.shape[-1]) if out.ndim >= 2 else out.reshape(1, -1)
        if flat.shape[1] < 2:
            raise RuntimeError("ONNX output does not contain mean/log_var values.")
        mean = flat[:, 0]
        log_var = flat[:, 1]
    else:
        raise RuntimeError("ONNX model returned no outputs.")

    mean_val = float(np.asarray(mean).reshape(-1)[0])
    log_var_val = float(np.asarray(log_var).reshape(-1)[0])
    return mean_val, log_var_val


def _extract_hands(
    result: vision.HandLandmarkerResult, width: int, height: int
) -> List[Dict[str, Any]]:
    hands: List[Dict[str, Any]] = []
    for idx, landmarks in enumerate(result.hand_landmarks or []):
        points = []
        for landmark in landmarks:
            x = max(0.0, min(1.0, float(landmark.x))) * width
            y = max(0.0, min(1.0, float(landmark.y))) * height
            points.append({"x": float(x), "y": float(y), "z": float(landmark.z)})

        handedness = None
        score = None
        if result.handedness and len(result.handedness) > idx:
            classification_list = result.handedness[idx]
            if classification_list:
                handed = classification_list[0]
                handedness = handed.category_name
                score = float(handed.score)

        hands.append(
            {"points": points, "handedness": handedness, "score": score}
        )
    return hands


def _segment_hands(
    rgb_image: np.ndarray, hands: List[Dict[str, Any]]
) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    if not hands:
        return np.zeros((height, width), dtype=np.uint8)

    predictor = _load_sam_predictor()
    predictor.set_image(rgb_image)
    combined = np.zeros((height, width), dtype=bool)

    with torch.no_grad():
        for hand in hands:
            coords = np.array(
                [[p["x"], p["y"]] for p in hand["points"]], dtype=np.float32
            )
            if coords.size == 0:
                continue
            labels = np.ones((coords.shape[0],), dtype=np.int32)
            masks, _scores, _logits = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=False,
            )
            mask = masks[0]
            combined = np.logical_or(combined, mask)

    return (combined.astype(np.uint8) * 255)


def _apply_mask(rgb_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape[:2] != rgb_image.shape[:2]:
        mask = np.array(
            Image.fromarray(mask, mode="L").resize(
                (rgb_image.shape[1], rgb_image.shape[0]), Image.NEAREST
            )
        )
    if mask.ndim == 2:
        mask = mask[:, :, None]
    mask_f = (mask.astype(np.float32) / 255.0)
    masked = (rgb_image.astype(np.float32) * mask_f).clip(0, 255)
    return masked.astype(np.uint8)


def _maybe_save_image(image: Image.Image) -> None:
    if not SAVE_MASKED_IMAGE:
        return
    os.makedirs(os.path.dirname(SAVE_MASKED_PATH), exist_ok=True)
    image.save(SAVE_MASKED_PATH)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        session, input_name, input_size = _load_age_model()
        job_input = job.get("input", {})
        img = _decode_image(job_input)
        rgb_image = np.array(img)

        landmarker = _load_hand_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = landmarker.detect(mp_image)
        hands = _extract_hands(result, img.width, img.height)

        mask = _segment_hands(rgb_image, hands)
        if hands and np.any(mask):
            rgb_image = _apply_mask(rgb_image, mask)

        masked_img = Image.fromarray(rgb_image)
        _maybe_save_image(masked_img)
        tensor = _prepare_image(masked_img, input_size)
        outputs = session.run(None, {input_name: tensor})
        mean_val, log_var_val = _extract_mean_logvar(outputs)
        std_val = float(np.exp(0.5 * log_var_val))
        return {"age": mean_val, "std": std_val}
    except Exception as exc:
        return {"error": str(exc)}


def _should_use_runpod() -> bool:
    if os.environ.get("FORCE_RUNPOD_SERVERLESS") == "1":
        return True
    if os.environ.get("RUNPOD_SERVERLESS") == "1":
        return True
    if os.environ.get("RUNPOD_ENDPOINT_ID"):
        return True
    if os.environ.get("RUNPOD_API_KEY"):
        return True
    return False


def _ensure_test_input() -> None:
    path = os.environ.get("RUNPOD_TEST_INPUT", "test_input.json")
    if os.path.exists(path):
        return
    img = Image.new("RGB", (1, 1), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = {"input": {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


class _LocalHandler(BaseHTTPRequestHandler):
    server_version = "local-age-regressor/2.0"

    def log_message(self, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path not in ("/", "/run", "/invocations"):
            self.send_error(404, "Not Found")
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON payload")
            return

        result = handler(payload)
        status = 200 if "error" not in result else 400
        body = json.dumps(result).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve_local() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    with HTTPServer((host, port), _LocalHandler) as httpd:
        httpd.serve_forever()


if _should_use_runpod():
    _ensure_test_input()
    runpod.serverless.start({"handler": handler})
else:
    _serve_local()
