import uuid

import sqlalchemy as sa
from sqlalchemy import Column, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    fname = Column(Text, nullable=False)
    lname = Column(Text, nullable=False)
    phone_number = Column(Text, nullable=False)
    birth_day = Column(DateTime(timezone=True), nullable=False)

    results = relationship("Result", back_populates="user", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    result_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    medical_report = Column(Text, nullable=False)
    normal_count = Column(sa.Integer(), server_default="0", nullable=False)
    abnormal_count = Column(sa.Integer(), server_default="0", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("User", back_populates="results")
    images = relationship("Image", back_populates="result", cascade="all, delete-orphan")


class Image(Base):
    __tablename__ = "images"

    image_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    result_id = Column(UUID(as_uuid=True), ForeignKey("results.result_id", ondelete="CASCADE"), nullable=False)
    org_img = Column(Text, nullable=False)
    result_img = Column(Text, nullable=False)

    result = relationship("Result", back_populates="images")
