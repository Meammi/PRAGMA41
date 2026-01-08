from datetime import date, datetime
from typing import List

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.repositories.images import create_image
from app.repositories.results import create_result
from app.repositories.users import get_or_create_user
from app.services.storage import decode_data_url, save_original, save_result_bytes


def parse_birth_day(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="birth_day must be ISO format (YYYY-MM-DD)") from exc
    if parsed.date() > date.today():
        raise HTTPException(status_code=400, detail="birth_day cannot be in the future")
    return parsed


def submit_case(
    db: Session,
    fname: str,
    lname: str,
    phone_number: str,
    birth_day: str,
    images,
    result_images: List[str],
) -> dict:
    if not images:
        raise HTTPException(status_code=400, detail="No images uploaded")
    if len(result_images) != len(images):
        raise HTTPException(status_code=400, detail="Result images do not match uploads")

    user = get_or_create_user(db, fname, lname, phone_number, parse_birth_day(birth_day))
    result = create_result(db, user.user_id)

    saved = []
    for index, image in enumerate(images):
        contents = image.file.read()
        if not contents:
            continue

        original_filename = image.filename or f"upload-{index}"
        org_img = save_original(contents, original_filename)

        result_bytes = decode_data_url(result_images[index])
        result_img = save_result_bytes(result_bytes)

        image_row = create_image(db, result.result_id, org_img, result_img)
        saved.append((index, image_row))

    if not saved:
        db.rollback()
        raise HTTPException(status_code=400, detail="No valid images were processed")

    db.commit()

    return {
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
