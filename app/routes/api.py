from typing import List

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.images import get_image
from app.services.cases import submit_case
from app.services.inference import detect_bytes
from app.services.storage import resolve_path

router = APIRouter()


@router.post("/detect")
async def detect(images: List[UploadFile] = File(...)) -> JSONResponse:
    payload = []
    for image in images:
        payload.append({"filename": image.filename, "content": await image.read()})

    results = detect_bytes(payload)
    return JSONResponse({"results": results})


@router.post("/submit")
async def submit(
    fname: str = Form(...),
    lname: str = Form(...),
    phone_number: str = Form(...),
    birth_day: str = Form(...),
    images: List[UploadFile] = File(...),
    result_images: List[str] = Form(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    data = submit_case(db, fname, lname, phone_number, birth_day, images, result_images)
    return JSONResponse(data)


@router.get("/images/{image_id}/original")
def download_original(image_id: str, db: Session = Depends(get_db)):
    image_row = get_image(db, image_id)
    if not image_row:
        return JSONResponse({"detail": "Image not found"}, status_code=404)
    file_path = resolve_path(image_row.org_img)
    if not file_path.exists():
        return JSONResponse({"detail": "File not found"}, status_code=404)
    return FileResponse(file_path, filename=file_path.name)


@router.get("/images/{image_id}/result")
def download_result(image_id: str, db: Session = Depends(get_db)):
    image_row = get_image(db, image_id)
    if not image_row:
        return JSONResponse({"detail": "Image not found"}, status_code=404)
    file_path = resolve_path(image_row.result_img)
    if not file_path.exists():
        return JSONResponse({"detail": "File not found"}, status_code=404)
    return FileResponse(file_path, filename=file_path.name)
