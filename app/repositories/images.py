from sqlalchemy.orm import Session

from app.models import Image


def create_image(db: Session, result_id, org_img: str, result_img: str):
    image = Image(
        result_id=result_id,
        org_img=org_img,
        result_img=result_img,
    )
    db.add(image)
    db.flush()
    return image


def get_image(db: Session, image_id: str):
    return db.query(Image).filter(Image.image_id == image_id).first()
