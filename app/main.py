import io
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.onnx"
PUBLIC_DIR = Path(__file__).resolve().parent.parent / "public"

INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7
CLASSES = ["cell"]

app = FastAPI()

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


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


@app.post("/detect")
async def detect(image: UploadFile = File(...)) -> StreamingResponse:
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    original_width, original_height = pil_image.size

    input_tensor = prepare_input(pil_image)
    outputs = session.run([output_name], {input_name: input_tensor})
    boxes = process_output(outputs[0], original_width, original_height)

    result_bytes = draw_boxes(pil_image.convert("RGB"), boxes)
    return StreamingResponse(io.BytesIO(result_bytes), media_type="image/jpeg")


app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="static")
