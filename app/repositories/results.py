from sqlalchemy.orm import Session

from app.models import Result


def create_result(db: Session, user_id, medical_report: str = "", normal_count: int = 0, abnormal_count: int = 0):
    result = Result(
        user_id=user_id,
        medical_report=medical_report,
        normal_count=normal_count,
        abnormal_count=abnormal_count,
    )
    db.add(result)
    db.flush()
    return result
