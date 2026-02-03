import base64
import io
from typing import Any, Dict, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
import runpod

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INPUT_SIZE = 480

_SESSION: ort.InferenceSession | None = None
_INPUT_NAME: str | None = None
_INPUT_SIZE: int | None = None


def _infer_img_size(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        if shape[2] == shape[3]:
            return int(shape[2])
    return None


def _load_model() -> Tuple[ort.InferenceSession, str, int]:
    global _SESSION, _INPUT_NAME, _INPUT_SIZE
    if _SESSION is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _SESSION = ort.InferenceSession("v2_m_age_regressor_ddp.onnx", providers=providers)
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


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        session, input_name, input_size = _load_model()
        job_input = job.get("input", {})
        img = _decode_image(job_input)
        tensor = _prepare_image(img, input_size)
        outputs = session.run(None, {input_name: tensor})
        mean_val, log_var_val = _extract_mean_logvar(outputs)
        std_val = float(np.exp(0.5 * log_var_val))
        return {"age": mean_val, "std": std_val}
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
