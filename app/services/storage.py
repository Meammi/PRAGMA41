import base64
import binascii
import os
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"
ORIGINALS_DIR = STORAGE_DIR / "originals"
RESULTS_DIR = STORAGE_DIR / "results"

ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_extension(filename: str, default_ext: str = ".jpg") -> str:
    _, ext = os.path.splitext(filename)
    if not ext:
        return default_ext
    return ext


def decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise HTTPException(status_code=400, detail="Invalid result image data")
    _, encoded = data_url.split(",", 1)
    try:
        return base64.b64decode(encoded)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid result image data") from exc


def save_original(contents: bytes, original_filename: str) -> str:
    original_ext = ensure_extension(original_filename)
    original_name = f"{uuid4().hex}{original_ext}"
    original_path = ORIGINALS_DIR / original_name
    original_path.write_bytes(contents)
    return str(original_path.relative_to(STORAGE_DIR))


def save_result_bytes(result_bytes: bytes) -> str:
    result_name = f"{uuid4().hex}.jpg"
    result_path = RESULTS_DIR / result_name
    result_path.write_bytes(result_bytes)
    return str(result_path.relative_to(STORAGE_DIR))


def resolve_path(relative_path: str) -> Path:
    return STORAGE_DIR / relative_path
