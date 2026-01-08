import re

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.models import User


def normalize_name(value: str) -> str:
    return value.strip().lower()


def normalize_phone(value: str) -> str:
    return re.sub(r"\D+", "", value)


def find_user(db: Session, fname: str, lname: str, phone_number: str):
    fname_norm = normalize_name(fname)
    lname_norm = normalize_name(lname)
    phone_norm = normalize_phone(phone_number)

    phone_expr = func.regexp_replace(User.phone_number, r"\\D+", "", "g")
    name_match = (
        func.lower(func.trim(User.fname)) == fname_norm,
        func.lower(func.trim(User.lname)) == lname_norm,
    )

    return db.query(User).filter(or_(and_(*name_match), phone_expr == phone_norm)).first()


def create_user(db: Session, fname: str, lname: str, phone_number: str, birth_day):
    user = User(
        fname=fname.strip(),
        lname=lname.strip(),
        phone_number=phone_number.strip(),
        birth_day=birth_day,
    )
    db.add(user)
    db.flush()
    return user


def get_or_create_user(db: Session, fname: str, lname: str, phone_number: str, birth_day):
    existing = find_user(db, fname, lname, phone_number)
    if existing:
        return existing
    return create_user(db, fname, lname, phone_number, birth_day)
