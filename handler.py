import base64
import fnmatch
import hashlib
import io
import json
import os
import shutil
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple

import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mobile_sam import SamPredictor, sam_model_registry
from PIL import Image, ImageOps
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
AGE_MODEL_PATH = os.environ.get("AGE_MODEL_PATH", "/app/v2_m_age_regressor_ddp.onnx")
AGE_MODEL_URL = os.environ.get(
    "AGE_MODEL_URL",
    "https://github.com/greenwich-xr-security/ONNX_RunPod_Serverless/releases/download/1/v2_m_age_regressor_ddp.onnx",
)
AGE_MODEL_SHA256 = os.environ.get(
    "AGE_MODEL_SHA256",
    "618a3935d3e5a15c9f7ec3f39fe759b2239497d5184b146a8c55dc6660add395",
)
MODEL_DOWNLOAD_TIMEOUT = int(os.environ.get("MODEL_DOWNLOAD_TIMEOUT", "1200"))

MODEL_REPO = os.environ.get("MODEL_REPO", "").strip()
MODEL_REPO_OWNER = os.environ.get("MODEL_REPO_OWNER", "").strip()
MODEL_REPO_NAME = os.environ.get("MODEL_REPO_NAME", "").strip()
if MODEL_REPO and not (MODEL_REPO_OWNER and MODEL_REPO_NAME):
    parts = MODEL_REPO.split("/", 1)
    if len(parts) == 2:
        MODEL_REPO_OWNER, MODEL_REPO_NAME = parts[0].strip(), parts[1].strip()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
MODEL_STORE_DIR = os.environ.get("MODEL_STORE_DIR", "/app/models")
MODEL_MANIFEST_PATH = os.path.join(MODEL_STORE_DIR, "manifest.json")
MODEL_ASSET_PATTERNS = [
    p.strip()
    for p in os.environ.get(
        "MODEL_ASSET_PATTERNS", "*.onnx,*.sha256,model_card.md,deployment_sheet.json"
    ).split(",")
    if p.strip()
]
MODEL_MAX_RELEASES = int(os.environ.get("MODEL_MAX_RELEASES", "0"))
MODEL_ALLOW_PRERELEASE = os.environ.get("MODEL_ALLOW_PRERELEASE", "1") == "1"
MODEL_ALLOW_DRAFT = os.environ.get("MODEL_ALLOW_DRAFT", "0") == "1"
MODEL_SYNC_MIN_INTERVAL_SEC = int(os.environ.get("MODEL_SYNC_MIN_INTERVAL_SEC", "300"))

SAVE_INFERENCE_LOGS = os.environ.get("SAVE_INFERENCE_LOGS", "0") == "1"
INFERENCE_LOG_JPEG_QUALITY = int(os.environ.get("INFERENCE_LOG_JPEG_QUALITY", "90"))
INFERENCE_LOG_STORAGE = os.environ.get("INFERENCE_LOG_STORAGE", "s3").strip().lower()
INFERENCE_LOG_S3_BUCKET = os.environ.get("INFERENCE_LOG_S3_BUCKET", "").strip()
INFERENCE_LOG_S3_PREFIX = os.environ.get("INFERENCE_LOG_S3_PREFIX", "inference_logs").strip(
    "/"
)
INFERENCE_LOG_S3_ENDPOINT_URL = os.environ.get("INFERENCE_LOG_S3_ENDPOINT_URL", "").strip()
INFERENCE_LOG_S3_REGION = os.environ.get("INFERENCE_LOG_S3_REGION", "us-east-1").strip()
INFERENCE_LOG_S3_ACCESS_KEY_ID = os.environ.get(
    "INFERENCE_LOG_S3_ACCESS_KEY_ID", ""
).strip()
INFERENCE_LOG_S3_SECRET_ACCESS_KEY = os.environ.get(
    "INFERENCE_LOG_S3_SECRET_ACCESS_KEY", ""
).strip()
MAX_HANDS = int(os.environ.get("MAX_HANDS", "2"))
MIN_HAND_DET_CONF = float(os.environ.get("MIN_HAND_DET_CONF", "0.5"))
MIN_HAND_PRESENCE_CONF = float(os.environ.get("MIN_HAND_PRESENCE_CONF", "0.5"))
MIN_HAND_TRACKING_CONF = float(os.environ.get("MIN_HAND_TRACKING_CONF", "0.5"))
HAND_CROP_BORDER_RATIO = float(os.environ.get("HAND_CROP_BORDER_RATIO", "0.05"))
if not 0.0 <= HAND_CROP_BORDER_RATIO < 0.5:
    raise ValueError("HAND_CROP_BORDER_RATIO must be in the range [0.0, 0.5).")

_HAND_LANDMARKER: vision.HandLandmarker | None = None
_SAM_PREDICTOR: SamPredictor | None = None
_DEVICE: str | None = None

_MODEL_SESSION_CACHE: Dict[str, Tuple[ort.InferenceSession, str, int]] = {}
_MODEL_SESSION_LOCK = threading.Lock()

_MODEL_CATALOG_CACHE: Dict[str, Any] | None = None
_MODEL_CATALOG_LOCK = threading.Lock()
_MODEL_CATALOG_LAST_SYNC = 0.0
_INFERENCE_LOG_LOCK = threading.Lock()
_INFERENCE_S3_CLIENT: Any | None = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_INPUT_SIZE = 480
GITHUB_API_BASE = "https://api.github.com"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mkdir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _get_running_mode() -> Any:
    if hasattr(vision, "RunningMode"):
        return vision.RunningMode.IMAGE
    if hasattr(vision, "VisionRunningMode"):
        return vision.VisionRunningMode.IMAGE
    raise RuntimeError("Unsupported mediapipe vision running mode.")


def _http_headers(include_github_json_accept: bool = False) -> Dict[str, str]:
    headers = {"User-Agent": "onnx-runpod-serverless"}
    if include_github_json_accept:
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _download_to_path(path: str, url: str, headers: Dict[str, str] | None = None) -> None:
    _mkdir_for_file(path)
    tmp_path = f"{path}.tmp"
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
            with open(tmp_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _ensure_file(
    path: str,
    url: str,
    sha256: str | None = None,
    headers: Dict[str, str] | None = None,
) -> None:
    if os.path.exists(path):
        return
    _download_to_path(path, url, headers=headers)
    if sha256:
        digest = _sha256_file(path)
        if digest.lower() != sha256.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {path}: expected {sha256}, got {digest}"
            )


def _http_get_json(url: str, headers: Dict[str, str]) -> Any:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _repo_configured() -> bool:
    return bool(MODEL_REPO_OWNER and MODEL_REPO_NAME)


def _release_dir(tag: str) -> str:
    safe_tag = tag.replace("/", "_")
    return os.path.join(MODEL_STORE_DIR, safe_tag)


def _asset_selected(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in MODEL_ASSET_PATTERNS)


def _parse_sha256_file(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    base_name = os.path.basename(path)
    fallback_name = base_name[:-7] if base_name.endswith(".sha256") else ""
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            digest = parts[0]
            name = fallback_name
            if len(parts) >= 2:
                name = parts[-1].lstrip("*")
            if name:
                mapping[os.path.basename(name)] = digest
    return mapping


def _read_text_if_exists(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _read_json_if_exists(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest_from_disk() -> Dict[str, Any] | None:
    if not os.path.exists(MODEL_MANIFEST_PATH):
        return None
    with open(MODEL_MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(manifest: Dict[str, Any]) -> None:
    _mkdir_for_file(MODEL_MANIFEST_PATH)
    tmp_path = f"{MODEL_MANIFEST_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(tmp_path, MODEL_MANIFEST_PATH)


def _is_runpod_runtime_env() -> bool:
    if os.environ.get("FORCE_RUNPOD_SERVERLESS") == "1":
        return True
    if os.environ.get("RUNPOD_SERVERLESS") == "1":
        return True
    if os.environ.get("RUNPOD_ENDPOINT_ID"):
        return True
    if os.environ.get("RUNPOD_API_KEY"):
        return True
    return False


def _inference_log_storage_mode() -> str:
    if INFERENCE_LOG_STORAGE != "s3":
        raise ValueError("INFERENCE_LOG_STORAGE must be set to: s3")
    return "s3"


def _inference_log_s3_key(name: str) -> str:
    if INFERENCE_LOG_S3_PREFIX:
        return f"{INFERENCE_LOG_S3_PREFIX}/{name}"
    return name


def _inference_log_index_ref() -> str:
    _inference_log_storage_mode()
    return _inference_log_s3_key("index.json")


def _get_inference_s3_client() -> Any:
    global _INFERENCE_S3_CLIENT
    if _INFERENCE_S3_CLIENT is not None:
        return _INFERENCE_S3_CLIENT
    if not INFERENCE_LOG_S3_BUCKET:
        raise ValueError("INFERENCE_LOG_S3_BUCKET is required for S3 log storage.")

    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 inference log storage.") from exc

    kwargs: Dict[str, Any] = {}
    if INFERENCE_LOG_S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = INFERENCE_LOG_S3_ENDPOINT_URL
    if INFERENCE_LOG_S3_REGION:
        kwargs["region_name"] = INFERENCE_LOG_S3_REGION
    if INFERENCE_LOG_S3_ACCESS_KEY_ID and INFERENCE_LOG_S3_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = INFERENCE_LOG_S3_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = INFERENCE_LOG_S3_SECRET_ACCESS_KEY

    _INFERENCE_S3_CLIENT = boto3.client("s3", **kwargs)
    return _INFERENCE_S3_CLIENT


def _s3_get_object_bytes(key: str) -> bytes | None:
    client = _get_inference_s3_client()
    try:
        response = client.get_object(Bucket=INFERENCE_LOG_S3_BUCKET, Key=key)
        return response["Body"].read()
    except Exception as exc:
        response = getattr(exc, "response", {}) or {}
        error = response.get("Error", {}) or {}
        code = str(error.get("Code", "")).strip()
        message = str(error.get("Message", "")).strip().lower()
        status = str(response.get("ResponseMetadata", {}).get("HTTPStatusCode", "")).strip()

        # Some S3-compatible providers return non-standard codes/messages for missing keys.
        if (
            code in {"NoSuchKey", "404", "NotFound", "InvalidArgument"}
            and "not found" in message
        ) or code in {"NoSuchKey", "404", "NotFound"} or status == "404":
            return None
        raise


def _s3_put_object_bytes(key: str, raw: bytes, content_type: str) -> None:
    client = _get_inference_s3_client()
    client.put_object(
        Bucket=INFERENCE_LOG_S3_BUCKET,
        Key=key,
        Body=raw,
        ContentType=content_type,
    )


def _s3_delete_object(key: str) -> None:
    client = _get_inference_s3_client()
    client.delete_object(Bucket=INFERENCE_LOG_S3_BUCKET, Key=key)


def _s3_list_objects(prefix: str) -> List[str]:
    client = _get_inference_s3_client()
    token: str | None = None
    keys: List[str] = []
    while True:
        params: Dict[str, Any] = {"Bucket": INFERENCE_LOG_S3_BUCKET, "Prefix": prefix}
        if token:
            params["ContinuationToken"] = token
        response = client.list_objects_v2(**params)
        contents = response.get("Contents", []) or []
        for item in contents:
            key = item.get("Key")
            if isinstance(key, str):
                keys.append(key)
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return keys


def _read_inference_log_index() -> Dict[str, Any]:
    _inference_log_storage_mode()
    raw = _s3_get_object_bytes(_inference_log_index_ref())
    if not raw:
        prefix = _inference_log_s3_key("")
        keys = _s3_list_objects(prefix)
        index_key = _inference_log_index_ref()
        ids: List[str] = []
        for key in keys:
            if key == index_key or not key.endswith(".json"):
                continue
            name = key.rsplit("/", 1)[-1]
            ids.append(name[:-5])
        ids.sort(reverse=True)
        return {"items": ids}
    data = json.loads(raw.decode("utf-8"))

    if not isinstance(data, dict):
        return {"items": []}
    items = data.get("items")
    if not isinstance(items, list):
        return {"items": []}
    return {"items": [str(item) for item in items]}


def _write_inference_log_index(index: Dict[str, Any]) -> None:
    _inference_log_storage_mode()
    _s3_put_object_bytes(
        _inference_log_index_ref(),
        json.dumps(index, indent=2).encode("utf-8"),
        "application/json",
    )


def _inference_log_paths(inference_id: str) -> Tuple[str, str]:
    _inference_log_storage_mode()
    image_ref = _inference_log_s3_key(f"{inference_id}.jpg")
    meta_ref = _inference_log_s3_key(f"{inference_id}.json")
    return image_ref, meta_ref


def _save_inference_log(
    img: Image.Image,
    model_tag: str | None,
    model_name: str | None,
    age: float,
    std: float,
) -> str | None:
    if not SAVE_INFERENCE_LOGS:
        return None
    _inference_log_storage_mode()
    inference_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    image_ref, meta_ref = _inference_log_paths(inference_id)
    metadata = {
        "id": inference_id,
        "created_at": _now_iso(),
        "model_tag": model_tag,
        "model_name": model_name,
        "age": age,
        "std": std,
        "storage": "s3",
        "image_key": image_ref,
        "metadata_key": meta_ref,
    }

    with _INFERENCE_LOG_LOCK:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=INFERENCE_LOG_JPEG_QUALITY)
        _s3_put_object_bytes(image_ref, buf.getvalue(), "image/jpeg")
        _s3_put_object_bytes(
            meta_ref,
            json.dumps(metadata, indent=2).encode("utf-8"),
            "application/json",
        )

        index = _read_inference_log_index()
        items = [inference_id] + [item for item in index["items"] if item != inference_id]
        index["items"] = items
        _write_inference_log_index(index)
    return inference_id


def _read_inference_log_metadata(inference_id: str) -> Dict[str, Any]:
    _inference_log_storage_mode()
    _image_ref, meta_ref = _inference_log_paths(inference_id)
    raw = _s3_get_object_bytes(meta_ref)
    if not raw:
        raise ValueError(f"Unknown inference_id: {inference_id}")
    return json.loads(raw.decode("utf-8"))


def _read_inference_log_image_bytes(inference_id: str, meta: Dict[str, Any]) -> bytes:
    _inference_log_storage_mode()
    image_key = str(meta.get("image_key", ""))
    if not image_key:
        image_key, _meta_key = _inference_log_paths(inference_id)
    if not image_key:
        raise ValueError(f"Image not found for inference_id: {inference_id}")
    raw = _s3_get_object_bytes(image_key)
    if not raw:
        raise ValueError(f"Image not found for inference_id: {inference_id}")
    return raw


def _list_inference_logs_payload(limit: int | None = None) -> Dict[str, Any]:
    if not SAVE_INFERENCE_LOGS:
        return {"enabled": False, "items": []}
    with _INFERENCE_LOG_LOCK:
        index = _read_inference_log_index()
        ids = index["items"]
        if limit is not None:
            ids = ids[: max(0, limit)]
        items: List[Dict[str, Any]] = []
        for inference_id in ids:
            try:
                meta = _read_inference_log_metadata(inference_id)
            except Exception:
                continue
            items.append(
                {
                    "id": meta.get("id"),
                    "created_at": meta.get("created_at"),
                    "model_tag": meta.get("model_tag"),
                    "model_name": meta.get("model_name"),
                    "age": meta.get("age"),
                    "std": meta.get("std"),
                }
            )
        return {
            "enabled": True,
            "storage": "s3",
            "index_ref": _inference_log_index_ref(),
            "count": len(items),
            "items": items,
        }


def _get_inference_log_payload(inference_id: str, include_image: bool = False) -> Dict[str, Any]:
    if not SAVE_INFERENCE_LOGS:
        raise ValueError("Inference log storage is disabled.")
    with _INFERENCE_LOG_LOCK:
        meta = _read_inference_log_metadata(inference_id)
        if include_image:
            raw = _read_inference_log_image_bytes(inference_id, meta)
            meta["image_base64"] = base64.b64encode(raw).decode("utf-8")
        return meta


def _fetch_releases() -> List[Dict[str, Any]]:
    if not _repo_configured():
        return []
    headers = _http_headers(include_github_json_accept=True)
    releases: List[Dict[str, Any]] = []
    page = 1
    while True:
        url = (
            f"{GITHUB_API_BASE}/repos/{MODEL_REPO_OWNER}/{MODEL_REPO_NAME}/releases"
            f"?per_page=100&page={page}"
        )
        batch = _http_get_json(url, headers)
        if not batch:
            break
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return releases


def _sync_release_assets(release: Dict[str, Any]) -> Dict[str, Any] | None:
    tag = release.get("tag_name") or ""
    if not tag:
        return None
    if release.get("draft") and not MODEL_ALLOW_DRAFT:
        return None
    if release.get("prerelease") and not MODEL_ALLOW_PRERELEASE:
        return None

    assets = release.get("assets") or []
    release_path = _release_dir(tag)
    os.makedirs(release_path, exist_ok=True)

    kept_assets: List[Dict[str, Any]] = []
    for asset in assets:
        name = asset.get("name") or ""
        url = asset.get("browser_download_url") or ""
        if not name or not url or not _asset_selected(name):
            continue
        local_path = os.path.join(release_path, name)
        if not os.path.exists(local_path):
            _download_to_path(local_path, url, headers=_http_headers())
        kept_assets.append(
            {
                "name": name,
                "size": int(asset.get("size") or 0),
                "content_type": asset.get("content_type"),
                "url": url,
                "path": local_path,
            }
        )

    sha_map: Dict[str, str] = {}
    for item in kept_assets:
        if item["name"].endswith(".sha256") and os.path.exists(item["path"]):
            sha_map.update(_parse_sha256_file(item["path"]))

    models: List[Dict[str, Any]] = []
    for item in kept_assets:
        if not item["name"].endswith(".onnx"):
            continue
        local_path = item["path"]
        digest = _sha256_file(local_path)
        expected = sha_map.get(item["name"])
        if expected and digest.lower() != expected.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {item['name']} in release {tag}: expected {expected}, got {digest}"
            )
        models.append(
            {
                "name": item["name"],
                "path": local_path,
                "size": item["size"],
                "sha256": digest,
            }
        )

    model_card_path = ""
    deployment_sheet_path = ""
    for item in kept_assets:
        lowered = item["name"].lower()
        if lowered == "model_card.md":
            model_card_path = item["path"]
        if lowered == "deployment_sheet.json":
            deployment_sheet_path = item["path"]

    model_card = _read_text_if_exists(model_card_path)
    deployment_sheet = _read_json_if_exists(deployment_sheet_path)

    return {
        "tag": tag,
        "name": release.get("name") or tag,
        "body": release.get("body") or "",
        "published_at": release.get("published_at"),
        "created_at": release.get("created_at"),
        "prerelease": bool(release.get("prerelease")),
        "draft": bool(release.get("draft")),
        "models": models,
        "assets": kept_assets,
        "model_card_path": model_card_path or None,
        "deployment_sheet_path": deployment_sheet_path or None,
        "model_card": model_card,
        "deployment_sheet": deployment_sheet,
    }


def _sync_model_catalog(force: bool = False) -> Dict[str, Any] | None:
    global _MODEL_CATALOG_CACHE, _MODEL_CATALOG_LAST_SYNC

    with _MODEL_CATALOG_LOCK:
        now = time.time()
        if not force and (now - _MODEL_CATALOG_LAST_SYNC) < MODEL_SYNC_MIN_INTERVAL_SEC:
            if _MODEL_CATALOG_CACHE is not None:
                return _MODEL_CATALOG_CACHE
            _MODEL_CATALOG_CACHE = _read_manifest_from_disk()
            return _MODEL_CATALOG_CACHE

        if not _repo_configured():
            _MODEL_CATALOG_CACHE = _read_manifest_from_disk()
            _MODEL_CATALOG_LAST_SYNC = now
            return _MODEL_CATALOG_CACHE

        releases = _fetch_releases()
        synced: List[Dict[str, Any]] = []
        for release in releases:
            entry = _sync_release_assets(release)
            if entry is None:
                continue
            if entry["models"]:
                synced.append(entry)

        if MODEL_MAX_RELEASES > 0:
            synced = synced[:MODEL_MAX_RELEASES]

        manifest = {
            "repo": f"{MODEL_REPO_OWNER}/{MODEL_REPO_NAME}",
            "synced_at": _now_iso(),
            "releases": synced,
            "default_tag": synced[0]["tag"] if synced else None,
        }
        _write_manifest(manifest)
        _MODEL_CATALOG_CACHE = manifest
        _MODEL_CATALOG_LAST_SYNC = now
        return manifest


def _get_catalog_cached_only() -> Dict[str, Any] | None:
    global _MODEL_CATALOG_CACHE
    if _MODEL_CATALOG_CACHE is not None:
        return _MODEL_CATALOG_CACHE
    with _MODEL_CATALOG_LOCK:
        if _MODEL_CATALOG_CACHE is not None:
            return _MODEL_CATALOG_CACHE
        _MODEL_CATALOG_CACHE = _read_manifest_from_disk()
        return _MODEL_CATALOG_CACHE


def _get_catalog_with_refresh() -> Dict[str, Any] | None:
    return _sync_model_catalog(force=False)


def _list_models_payload() -> Dict[str, Any]:
    catalog = _get_catalog_with_refresh()
    if not catalog:
        return {"models": [], "repo": None, "synced_at": None}

    models: List[Dict[str, Any]] = []
    for release in catalog.get("releases", []):
        models.append(
            {
                "tag": release.get("tag"),
                "name": release.get("name"),
                "published_at": release.get("published_at"),
                "model_count": len(release.get("models") or []),
                "models": [m.get("name") for m in release.get("models") or []],
            }
        )
    return {
        "repo": catalog.get("repo"),
        "synced_at": catalog.get("synced_at"),
        "default_tag": catalog.get("default_tag"),
        "models": models,
    }


def _get_model_payload(tag: str) -> Dict[str, Any]:
    catalog = _get_catalog_with_refresh()
    if not catalog:
        raise ValueError("Model catalog is not available.")
    for release in catalog.get("releases", []):
        if release.get("tag") == tag:
            return release
    raise ValueError(f"Unknown model tag: {tag}")


def _resolve_selected_model(
    model_tag: str | None,
    model_name: str | None,
) -> Tuple[str, str | None, str | None]:
    catalog = _get_catalog_cached_only()
    if catalog and catalog.get("releases"):
        releases = catalog["releases"]
        selected_release = releases[0]
        if model_tag and model_tag.lower() != "latest":
            selected_release = next(
                (release for release in releases if release.get("tag") == model_tag), None
            )
            if selected_release is None:
                raise ValueError(f"Unknown model_tag: {model_tag}")

        selected_model = None
        for model in selected_release.get("models") or []:
            if not model_name or model.get("name") == model_name:
                selected_model = model
                break
        if selected_model is None:
            if model_name:
                raise ValueError(
                    f"Model '{model_name}' not found in release '{selected_release.get('tag')}'"
                )
            raise ValueError(f"No model assets found in release '{selected_release.get('tag')}'")

        return (
            selected_model["path"],
            selected_release.get("tag"),
            selected_model.get("name"),
        )

    if not os.path.exists(AGE_MODEL_PATH):
        _ensure_file(
            AGE_MODEL_PATH,
            AGE_MODEL_URL,
            AGE_MODEL_SHA256,
            headers=_http_headers(),
        )
    return AGE_MODEL_PATH, None, os.path.basename(AGE_MODEL_PATH)


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


def _load_age_model(model_path: str) -> Tuple[ort.InferenceSession, str, int]:
    with _MODEL_SESSION_LOCK:
        cached = _MODEL_SESSION_CACHE.get(model_path)
        if cached is not None:
            return cached
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        input_size = _infer_img_size(session) or DEFAULT_INPUT_SIZE
        data = (session, input_name, input_size)
        _MODEL_SESSION_CACHE[model_path] = data
        return data


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


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"Invalid boolean value: {value}")


def _center_crop_frame(img: Image.Image) -> Image.Image:
    width, height = img.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    return img.crop((left, top, left + crop, top + crop))


def _prepare_image(img: Image.Image, size: int) -> np.ndarray:
    img = img.resize((size, size), Image.BILINEAR)
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
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    for idx, landmarks in enumerate(result.hand_landmarks or []):
        points = []
        for landmark in landmarks:
            x_norm = float(landmark.x)
            y_norm = float(landmark.y)
            x = (
                max(0.0, min(1.0, x_norm)) * max_x
                if np.isfinite(x_norm)
                else x_norm
            )
            y = (
                max(0.0, min(1.0, y_norm)) * max_y
                if np.isfinite(y_norm)
                else y_norm
            )
            points.append({"x": float(x), "y": float(y), "z": float(landmark.z)})

        handedness = None
        score = None
        if result.handedness and len(result.handedness) > idx:
            classification_list = result.handedness[idx]
            if classification_list:
                handed = classification_list[0]
                handedness = handed.category_name
                score = float(handed.score)

        hands.append({"points": points, "handedness": handedness, "score": score})
    return hands


def _hand_bbox(
    hand: Dict[str, Any], width: int, height: int
) -> Tuple[int, int, int, int] | None:
    if width <= 0 or height <= 0:
        return None

    points = hand.get("points")
    if not isinstance(points, list) or len(points) != 21:
        return None

    xs: List[float] = []
    ys: List[float] = []
    max_x = float(width - 1)
    max_y = float(height - 1)
    for point in points:
        if not isinstance(point, dict):
            return None
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(x) or not np.isfinite(y):
            return None
        xs.append(max(0.0, min(max_x, x)))
        ys.append(max(0.0, min(max_y, y)))

    xmin = int(np.floor(min(xs)))
    ymin = int(np.floor(min(ys)))
    xmax = int(np.ceil(max(xs)))
    ymax = int(np.ceil(max(ys)))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _select_primary_hand(
    hands: List[Dict[str, Any]], width: int, height: int
) -> Tuple[Dict[str, Any] | None, Tuple[int, int, int, int] | None]:
    primary_hand = None
    primary_bbox = None
    primary_area = -1

    for hand in hands:
        bbox = _hand_bbox(hand, width, height)
        if bbox is None:
            continue
        xmin, ymin, xmax, ymax = bbox
        area = (xmax - xmin) * (ymax - ymin)
        if area > primary_area:
            primary_hand = hand
            primary_bbox = bbox
            primary_area = area

    return primary_hand, primary_bbox


def _make_square_bbox(
    bbox: Tuple[int, int, int, int],
    border_ratio: float = HAND_CROP_BORDER_RATIO,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = [int(value) for value in bbox]
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid hand bounding box: {bbox}")
    if not 0.0 <= border_ratio < 0.5:
        raise ValueError("border_ratio must be in the range [0.0, 0.5).")

    hand_side = max(xmax - xmin, ymax - ymin)
    side = int(np.ceil(hand_side / (1.0 - 2.0 * border_ratio)))
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    square_xmin = int(np.floor(center_x - side / 2.0))
    square_ymin = int(np.floor(center_y - side / 2.0))
    return (
        square_xmin,
        square_ymin,
        square_xmin + side,
        square_ymin + side,
    )


def _crop_with_padding(
    img: Image.Image, square_bbox: Tuple[int, int, int, int]
) -> Image.Image:
    square_xmin, square_ymin, square_xmax, square_ymax = square_bbox
    if (
        square_xmax <= square_xmin
        or square_ymax <= square_ymin
        or square_xmax - square_xmin != square_ymax - square_ymin
    ):
        raise ValueError(f"Invalid square bounding box: {square_bbox}")

    width, height = img.size
    pad_left = max(0, -square_xmin)
    pad_top = max(0, -square_ymin)
    pad_right = max(0, square_xmax - width)
    pad_bottom = max(0, square_ymax - height)

    if pad_left or pad_top or pad_right or pad_bottom:
        img = ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0),
        )
        square_xmin += pad_left
        square_xmax += pad_left
        square_ymin += pad_top
        square_ymax += pad_top

    square_xmin = max(0, square_xmin)
    square_ymin = max(0, square_ymin)
    square_xmax = max(square_xmin + 1, min(img.size[0], square_xmax))
    square_ymax = max(square_ymin + 1, min(img.size[1], square_ymax))
    cropped = img.crop((square_xmin, square_ymin, square_xmax, square_ymax))
    if cropped.width != cropped.height:
        raise RuntimeError(
            f"Hand-centred crop is not square: {cropped.width}x{cropped.height}"
        )
    return cropped


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


def _run_inference(job_input: Dict[str, Any]) -> Dict[str, Any]:
    model_tag = job_input.get("model_tag")
    model_name = job_input.get("model_name")
    model_path, selected_tag, selected_model_name = _resolve_selected_model(
        model_tag=model_tag,
        model_name=model_name,
    )

    img = _decode_image(job_input)
    rgb_image = np.array(img)

    use_hand_landmarks = _coerce_bool(job_input.get("use_hand_landmarks"), default=True)
    use_hand_masking = _coerce_bool(job_input.get("use_hand_masking"), default=True)
    if use_hand_masking and not use_hand_landmarks:
        raise ValueError("use_hand_masking=true requires use_hand_landmarks=true")

    hands: List[Dict[str, Any]] = []
    primary_hand = None
    primary_bbox = None
    if use_hand_landmarks:
        landmarker = _load_hand_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = landmarker.detect(mp_image)
        hands = _extract_hands(result, img.width, img.height)
        primary_hand, primary_bbox = _select_primary_hand(
            hands, img.width, img.height
        )
        if primary_hand is None or primary_bbox is None:
            return {
                "age": -1.0,
                "std": 0.0,
                "model_tag": selected_tag,
                "model_name": selected_model_name,
                "inference_id": None,
                "use_hand_landmarks": use_hand_landmarks,
                "use_hand_masking": use_hand_masking,
            }
        hands = [primary_hand]

    if use_hand_masking:
        mask = _segment_hands(rgb_image, hands)
        if hands and np.any(mask):
            rgb_image = _apply_mask(rgb_image, mask)

    processed_img = Image.fromarray(rgb_image)
    if primary_bbox is not None:
        processed_img = _crop_with_padding(
            processed_img, _make_square_bbox(primary_bbox)
        )
    else:
        processed_img = _center_crop_frame(processed_img)

    session, input_name, input_size = _load_age_model(model_path)
    tensor = _prepare_image(processed_img, input_size)
    outputs = session.run(None, {input_name: tensor})
    mean_val, log_var_val = _extract_mean_logvar(outputs)
    std_val = float(np.exp(0.5 * log_var_val))
    inference_id = None
    inference_log_error = None
    try:
        inference_id = _save_inference_log(
            img=processed_img,
            model_tag=selected_tag,
            model_name=selected_model_name,
            age=mean_val,
            std=std_val,
        )
    except Exception as exc:
        inference_log_error = str(exc)

    response = {
        "age": mean_val,
        "std": std_val,
        "model_tag": selected_tag,
        "model_name": selected_model_name,
        "inference_id": inference_id,
        "use_hand_landmarks": use_hand_landmarks,
        "use_hand_masking": use_hand_masking,
    }
    if inference_log_error:
        response["inference_log_error"] = inference_log_error
    return response


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        job_input = job.get("input", {})
        action = str(job_input.get("action", "infer")).lower()

        if action == "list_models":
            return _list_models_payload()
        if action == "get_model":
            tag = str(job_input.get("model_tag", "")).strip()
            if not tag:
                raise ValueError("Missing required field: model_tag")
            return _get_model_payload(tag)
        if action == "list_inference_logs":
            limit_raw = job_input.get("limit")
            limit = int(limit_raw) if limit_raw is not None else None
            return _list_inference_logs_payload(limit=limit)
        if action == "get_inference_log":
            inference_id = str(job_input.get("inference_id", "")).strip()
            if not inference_id:
                raise ValueError("Missing required field: inference_id")
            include_image = bool(job_input.get("include_image", False))
            return _get_inference_log_payload(
                inference_id=inference_id,
                include_image=include_image,
            )

        return _run_inference(job_input)
    except Exception as exc:
        return {"error": str(exc)}


def _should_use_runpod() -> bool:
    return _is_runpod_runtime_env()


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
    server_version = "local-age-regressor/3.0"

    def log_message(self, *_args: Any) -> None:
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/models":
            self._send_json(200, _list_models_payload())
            return
        if path.startswith("/models/"):
            tag = path.split("/", 2)[-1]
            try:
                payload = _get_model_payload(tag)
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(404, {"error": str(exc)})
            return
        if path == "/inference-logs":
            self._send_json(200, _list_inference_logs_payload())
            return
        if path.startswith("/inference-logs/"):
            parts = path.split("/")
            if len(parts) >= 3:
                inference_id = parts[2]
                if len(parts) >= 4 and parts[3] == "image":
                    try:
                        payload = _get_inference_log_payload(
                            inference_id=inference_id,
                            include_image=False,
                        )
                        raw = _read_inference_log_image_bytes(inference_id, payload)
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(raw)))
                        self.end_headers()
                        self.wfile.write(raw)
                    except Exception as exc:
                        self._send_json(404, {"error": str(exc)})
                    return
                try:
                    payload = _get_inference_log_payload(
                        inference_id=inference_id,
                        include_image=False,
                    )
                    self._send_json(200, payload)
                except Exception as exc:
                    self._send_json(404, {"error": str(exc)})
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
        self._send_json(status, result)


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
