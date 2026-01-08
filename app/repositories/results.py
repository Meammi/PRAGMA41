from sqlalchemy.orm import Session

from app.models import Result


def create_result(db: Session, user_id, medical_report: str = ""):
    result = Result(
        user_id=user_id,
        medical_report=medical_report,
    )
    db.add(result)
    db.flush()
    return result
