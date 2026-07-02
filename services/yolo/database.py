import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base

DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
DB_PATH = "predictions.db"

engine = None
SessionLocal = None


def _build_database_url():
    if DB_BACKEND == "postgres":
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST", "localhost")
        db_name = os.getenv("DB_NAME", "yolo")
        return f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}"
    return f"sqlite:///{DB_PATH}"


def _configure_engine():
    global engine, SessionLocal

    database_url = _build_database_url()
    if DB_BACKEND == "postgres":
        engine = create_engine(database_url)
    else:
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    _configure_engine()
    Base.metadata.create_all(bind=engine)
