import base64
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CLASS_NAMES: Dict[int, str] = {0: "Normal", 1: "Abnormal"}

MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "model.onnx")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def prepare_input(image: Image.Image) -> Tuple[np.ndarray, float, int, int]:
    img = image.convert("RGB")
    scale = min(INPUT_SIZE / img.width, INPUT_SIZE / img.height)
    new_width = int(round(img.width * scale))
    new_height = int(round(img.height * scale))

    x_offset = int((INPUT_SIZE - new_width) // 2)
    y_offset = int((INPUT_SIZE - new_height) // 2)

    canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (0, 0, 0))
    resized = img.resize((new_width, new_height), Image.LANCZOS)
    canvas.paste(resized, (x_offset, y_offset))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(array, (2, 0, 1))
    return np.expand_dims(chw, axis=0), scale, x_offset, y_offset


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def label(self) -> str:
        return CLASS_NAMES.get(self.class_id, str(self.class_id))

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0 else 0.0


def _as_chw8400(output: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(output)
    if squeezed.shape == (6, 8400):
        return squeezed
    if squeezed.shape == (8400, 6):
        return squeezed.T
    if squeezed.shape in {(5, 8400), (8400, 5)}:
        raise ValueError(f"Model output seems to be 5 channels (old 1-class model): {squeezed.shape}")
    raise ValueError(f"Unexpected output shape: {squeezed.shape}")


def decode_yolo_output(output: np.ndarray, conf_threshold: float = CONFIDENCE_THRESHOLD) -> List[BoundingBox]:
    data = _as_chw8400(output)  # (6, 8400)
    boxes: List[BoundingBox] = []

    for i in range(data.shape[1]):
        cx = float(data[0, i])
        cy = float(data[1, i])
        w = float(data[2, i])
        h = float(data[3, i])

        score0 = float(data[4, i])
        score1 = float(data[5, i])

        if (score0 < 0.0 or score0 > 1.0) or (score1 < 0.0 or score1 > 1.0):
            score0 = sigmoid(score0)
            score1 = sigmoid(score1)

        if abs(cx) < 1.0 and abs(cy) < 1.0 and abs(w) < 1.0 and abs(h) < 1.0:
            cx *= INPUT_SIZE
            cy *= INPUT_SIZE
            w *= INPUT_SIZE
            h *= INPUT_SIZE

        confidence = score1 if score1 > score0 else score0
        class_id = 1 if score1 > score0 else 0
        if confidence < conf_threshold:
            continue

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        x1 = max(0.0, min(float(INPUT_SIZE), x1))
        y1 = max(0.0, min(float(INPUT_SIZE), y1))
        x2 = max(0.0, min(float(INPUT_SIZE), x2))
        y2 = max(0.0, min(float(INPUT_SIZE), y2))

        boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence, class_id=class_id))

    return boxes


def non_max_suppression(boxes: List[BoundingBox], iou_threshold: float = IOU_THRESHOLD) -> List[BoundingBox]:
    dets = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    selected: List[BoundingBox] = []

    for det in dets:
        if any((det.class_id == s.class_id and iou_xyxy(det.bbox, s.bbox) > iou_threshold) for s in selected):
            continue
        selected.append(det)
    return selected


def scale_boxes(
    boxes: List[BoundingBox],
    scale: float,
    x_offset: int,
    y_offset: int,
    original_width: int,
    original_height: int,
) -> List[BoundingBox]:
    scaled: List[BoundingBox] = []
    for det in boxes:
        x1 = (det.x1 - x_offset) / scale
        y1 = (det.y1 - y_offset) / scale
        x2 = (det.x2 - x_offset) / scale
        y2 = (det.y2 - y_offset) / scale

        x1 = max(0.0, min(float(original_width), x1))
        y1 = max(0.0, min(float(original_height), y1))
        x2 = max(0.0, min(float(original_width), x2))
        y2 = max(0.0, min(float(original_height), y2))

        scaled.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=det.confidence, class_id=det.class_id))
    return scaled


def _coerce_hex_color(color: str) -> str:
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        return color
    return "#000000"


def _contrast_text_color(bg_hex: str) -> str:
    # Simple luminance check; default to white text for dark-ish fills.
    bg_hex = _coerce_hex_color(bg_hex)
    r = int(bg_hex[1:3], 16)
    g = int(bg_hex[3:5], 16)
    b = int(bg_hex[5:7], 16)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 180 else "#FFFFFF"


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Prefer textbbox (more accurate); fall back to older APIs.
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    if hasattr(draw, "textsize"):
        w, h = draw.textsize(text, font=font)
        return int(w), int(h)
    try:
        w, h = font.getsize(text)  # type: ignore[attr-defined]
        return int(w), int(h)
    except Exception:
        return (len(text) * 6, 11)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y: float,
    text: str,
    bg_color: str,
    font: ImageFont.ImageFont,
    padding: int = 3,
    radius: int = 4,
) -> None:
    bg_color = _coerce_hex_color(bg_color)
    text_color = _contrast_text_color(bg_color)

    tw, th = _measure_text(draw, text, font)
    left = int(round(x))
    top = int(round(y))
    right = left + tw + padding * 2
    bottom = top + th + padding * 2

    # Background (rounded if available)
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle([left, top, right, bottom], radius=radius, fill=bg_color)
    else:
        draw.rectangle([left, top, right, bottom], fill=bg_color)

    # Text with subtle outline for extra contrast on busy backgrounds
    tx = left + padding
    ty = top + padding
    outline = "#000000" if text_color == "#FFFFFF" else "#FFFFFF"
    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((tx + ox, ty + oy), text, fill=outline, font=font)
    draw.text((tx, ty), text, fill=text_color, font=font)


def draw_boxes(image: Image.Image, boxes: List[BoundingBox]) -> bytes:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for box in boxes:
        color = "#4caf50" if box.class_id == 0 else "#f44336"
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=3)
        label = f"{box.label} {(box.confidence * 100.0):.1f}%"
        if font:
            # Place label above box if possible; otherwise place inside box.
            tw, th = _measure_text(draw, label, font)
            y_above = int(round(box.y1)) - (th + 8)
            y = y_above if y_above >= 0 else int(round(box.y1)) + 4
            x = int(round(box.x1)) + 4
            _draw_label(draw, x=x, y=y, text=label, bg_color=color, font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return buffer.read()


def detect_bytes(images: List[dict]) -> List[dict]:
    results = []
    for index, image in enumerate(images):
        contents = image["content"]
        filename = image.get("filename")
        pil_image = Image.open(io.BytesIO(contents))
        original_width, original_height = pil_image.size

        input_tensor, scale, x_offset, y_offset = prepare_input(pil_image)
        outputs = session.run([output_name], {input_name: input_tensor})
        dets_640 = decode_yolo_output(outputs[0], conf_threshold=CONFIDENCE_THRESHOLD)
        dets_nms = non_max_suppression(dets_640, iou_threshold=IOU_THRESHOLD)
        boxes = scale_boxes(dets_nms, scale, x_offset, y_offset, original_width, original_height)

        result_bytes = draw_boxes(pil_image.convert("RGB"), boxes)
        encoded = base64.b64encode(result_bytes).decode("ascii")
        results.append(
            {
                "index": index,
                "filename": filename or f"image-{index}",
                "result_image": f"data:image/jpeg;base64,{encoded}",
            }
        )
    return results
