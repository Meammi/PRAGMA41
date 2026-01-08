import base64
import binascii
import io
import os
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from .database import get_db
from .models import Image as ImageModel
from .models import Result, User

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.onnx"
PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
ORIGINALS_DIR = STORAGE_DIR / "originals"
RESULTS_DIR = STORAGE_DIR / "results"

INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
CLASSES = ["cell"]

app = FastAPI()

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_input(image: Image.Image) -> np.ndarray:
    resized = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def iou(box_a: "BoundingBox", box_b: "BoundingBox") -> float:
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)

    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


class BoundingBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, confidence: float, label: str) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence
        self.label = label


def process_output(output: np.ndarray, original_width: int, original_height: int) -> List[BoundingBox]:
    squeezed = np.squeeze(output)
    if squeezed.shape == (8400, 5):
        squeezed = squeezed.T

    if squeezed.shape != (5, 8400):
        raise ValueError(f"Unexpected output shape: {squeezed.shape}")

    boxes: List[BoundingBox] = []
    for idx in range(8400):
        confidence = float(squeezed[4, idx])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        xc = float(squeezed[0, idx])
        yc = float(squeezed[1, idx])
        w = float(squeezed[2, idx])
        h = float(squeezed[3, idx])

        x1 = (xc - w / 2) / INPUT_SIZE * original_width
        y1 = (yc - h / 2) / INPUT_SIZE * original_height
        x2 = (xc + w / 2) / INPUT_SIZE * original_width
        y2 = (yc + h / 2) / INPUT_SIZE * original_height

        boxes.append(BoundingBox(x1, y1, x2, y2, confidence, CLASSES[0]))

    boxes.sort(key=lambda b: b.confidence)
    merged: List[BoundingBox] = []

    for candidate in boxes:
        if any(iou(candidate, existing) > IOU_THRESHOLD for existing in merged):
            continue
        merged.append(candidate)

    return merged


def draw_boxes(image: Image.Image, boxes: List[BoundingBox]) -> bytes:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline="red", width=2)
        label = f"{box.label} ({box.confidence:.2f})"
        if font:
            text_origin = (box.x1 + 4, max(0, box.y1 - 12))
            draw.text(text_origin, label, fill="blue", font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return buffer.read()


def parse_birth_day(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="birth_day must be ISO format (YYYY-MM-DD)") from exc


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


@app.post("/detect")
async def detect(images: List[UploadFile] = File(...)) -> JSONResponse:
    results = []

    for index, image in enumerate(images):
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        original_width, original_height = pil_image.size

        input_tensor = prepare_input(pil_image)
        outputs = session.run([output_name], {input_name: input_tensor})
        boxes = process_output(outputs[0], original_width, original_height)

        result_bytes = draw_boxes(pil_image.convert("RGB"), boxes)
        encoded = base64.b64encode(result_bytes).decode("ascii")
        results.append(
            {
                "index": index,
                "filename": image.filename or f"image-{index}",
                "result_image": f"data:image/jpeg;base64,{encoded}",
            }
        )

    return JSONResponse({"results": results})


@app.post("/submit")
async def submit(
    fname: str = Form(...),
    lname: str = Form(...),
    phone_number: str = Form(...),
    birth_day: str = Form(...),
    images: List[UploadFile] = File(...),
    result_images: List[str] = Form(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    if len(result_images) != len(images):
        raise HTTPException(status_code=400, detail="Result images do not match uploads")

    user = User(
        fname=fname,
        lname=lname,
        phone_number=phone_number,
        birth_day=parse_birth_day(birth_day),
    )
    db.add(user)
    db.flush()

    result = Result(
        user_id=user.user_id,
        medical_report="",
    )
    db.add(result)
    db.flush()

    saved = []
    for index, image in enumerate(images):
        contents = await image.read()
        if not contents:
            continue

        original_filename = image.filename or f"upload-{uuid4().hex}"
        original_ext = ensure_extension(original_filename)
        original_name = f"{uuid4().hex}{original_ext}"
        original_path = ORIGINALS_DIR / original_name
        original_path.write_bytes(contents)

        result_bytes = decode_data_url(result_images[index])
        result_name = f"{uuid4().hex}.jpg"
        result_path = RESULTS_DIR / result_name
        result_path.write_bytes(result_bytes)

        image_row = ImageModel(
            result_id=result.result_id,
            org_img=str(original_path.relative_to(STORAGE_DIR)),
            result_img=str(result_path.relative_to(STORAGE_DIR)),
        )
        db.add(image_row)
        db.flush()
        saved.append((index, image_row))

    if not saved:
        db.rollback()
        raise HTTPException(status_code=400, detail="No valid images were processed")

    db.commit()

    return JSONResponse(
        {
            "user_id": str(user.user_id),
            "result_id": str(result.result_id),
            "images": [
                {
                    "index": index,
                    "image_id": str(row.image_id),
                    "org_img": row.org_img,
                    "result_img": row.result_img,
                }
                for index, row in saved
            ],
        }
    )


@app.get("/images/{image_id}/original")
def download_original(image_id: str, db: Session = Depends(get_db)):
    image_row = db.query(ImageModel).filter(ImageModel.image_id == image_id).first()
    if not image_row:
        raise HTTPException(status_code=404, detail="Image not found")
    file_path = STORAGE_DIR / image_row.org_img
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_path.name)


@app.get("/images/{image_id}/result")
def download_result(image_id: str, db: Session = Depends(get_db)):
    image_row = db.query(ImageModel).filter(ImageModel.image_id == image_id).first()
    if not image_row:
        raise HTTPException(status_code=404, detail="Image not found")
    file_path = STORAGE_DIR / image_row.result_img
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_path.name)


app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
