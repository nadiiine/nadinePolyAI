from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"

    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, server_default=func.current_timestamp())
    original_image = Column(String)
    predicted_image = Column(String)
    detections = relationship(
        "DetectionObject",
        back_populates="session",
        cascade="all, delete-orphan",
    )


class DetectionObject(Base):
    __tablename__ = "detection_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)
    session = relationship("PredictionSession", back_populates="detections")

    __table_args__ = (
        Index("idx_prediction_uid", "prediction_uid"),
        Index("idx_label", "label"),
        Index("idx_score", "score"),
    )
