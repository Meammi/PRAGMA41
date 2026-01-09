import io
import zipfile
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.images import get_image
from app.models import Result, User
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
    normal_count: int = Form(...),
    abnormal_count: int = Form(...),
    images: List[UploadFile] = File(...),
    result_images: List[str] = Form(...),
    db: Session = Depends(get_db),
) -> JSONResponse:
    data = submit_case(
        db,
        fname,
        lname,
        phone_number,
        birth_day,
        images,
        result_images,
        normal_count,
        abnormal_count,
    )
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


@router.get("/api/patients")
def list_patients(db: Session = Depends(get_db)) -> JSONResponse:
    users = db.query(User).order_by(User.lname.asc(), User.fname.asc()).all()
    response = []
    for user in users:
        results = sorted(user.results, key=lambda r: r.created_at or 0, reverse=True)
        response.append(
            {
                "user_id": str(user.user_id),
                "name": f"{user.fname} {user.lname}",
                "phone_number": user.phone_number,
                "birth_day": user.birth_day.isoformat(),
                "results": [
                    {
                        "result_id": str(result.result_id),
                        "created_at": result.created_at.isoformat() if result.created_at else None,
                        "image_count": len(result.images),
                        "normal_count": result.normal_count,
                        "abnormal_count": result.abnormal_count,
                    }
                    for result in results
                ],
            }
        )
    return JSONResponse({"patients": response})


@router.get("/api/results/{result_id}/zip")
def download_result_zip(result_id: str, db: Session = Depends(get_db)):
    result = db.query(Result).filter(Result.result_id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image in result.images:
            original_path = resolve_path(image.org_img)
            result_path = resolve_path(image.result_img)
            if original_path.exists():
                zf.write(original_path, arcname=f"originals/{original_path.name}")
            if result_path.exists():
                zf.write(result_path, arcname=f"results/{result_path.name}")

    buffer.seek(0)
    filename = f"result-{result_id}.zip"
    return StreamingResponse(buffer, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={filename}"})
