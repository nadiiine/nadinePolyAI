# YOLO API Data Layer Skill
 
This skill standardizes all database-related work in the YOLO service. It governs the migration from raw SQLite to SQLAlchemy and applies to all future database-layer changes: adding models, modifying schemas, creating database-backed endpoints, or switching the database backend.
 
---
 
## Activation Triggers
 
Use this skill for requests such as:
 
- Refactor the API to use SQLAlchemy
- Add a new SQLAlchemy model
- Add or remove database columns
- Add database-backed endpoints
- Delete prediction sessions
- Modify database queries
- Make the backend configurable for PostgreSQL
- Write tests for database-backed endpoints
- Fix violations of the project's database architecture
**Do not use** for frontend work, deployment, CI/CD, model inference changes, UI updates, or image-processing logic that does not affect persistence.
 
---
 
## Execution Workflow
 
### 1. Inspect the Current Implementation
 
Before modifying any code, locate every database access point:
 
- `sqlite3` imports
- `sqlite3.connect()` calls
- `cursor.execute()` calls
- Raw SQL strings
- Database initialization functions
- Every endpoint that reads or writes prediction data
> Do not assume all database code lives in a single file.
 
---
 
### 2. Preserve the Existing API Contract
 
The refactor must not change any of the following:
 
- Endpoint paths
- HTTP methods
- Request bodies
- Response JSON shape
- Response status codes
- Error messages
- Business logic
- YOLO inference behavior
Only the persistence implementation changes.
 
---
 
### 3. Build the SQLAlchemy Architecture
 
#### `database.py`
 
Responsibilities:
- Configure the database backend
- Create the SQLAlchemy engine
- Configure `SessionLocal`
- Expose `get_db()`
- Initialize tables via `Base.metadata.create_all()`
Database selection rules:
- SQLite by default
- PostgreSQL when `DB_BACKEND=postgres`
Use `connect_args={"check_same_thread": False}` **only** for SQLite.
 
```python
# database.py — reference structure
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
 
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")
 
if DB_BACKEND == "postgres":
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "yolo")
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)
else:
    DATABASE_URL = "sqlite:///./yolo.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
 
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
 
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
def init_db():
    Base.metadata.create_all(bind=engine)
```
 
---
 
#### `models.py`
 
Define all SQLAlchemy ORM models here. This file is the single source of truth for the schema.
 
Required models at minimum:
 
- `Base`
- `PredictionSession`
- `DetectionObject`
Map them to the existing tables and preserve the current schema exactly.
 
```python
# models.py — reference structure
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, relationship
import datetime
 
Base = declarative_base()
 
class PredictionSession(Base):
    __tablename__ = "prediction_sessions"
 
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    detections = relationship("DetectionObject", back_populates="session", cascade="all, delete-orphan")
 
class DetectionObject(Base):
    __tablename__ = "detection_objects"
 
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("prediction_sessions.id"), nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    session = relationship("PredictionSession", back_populates="detections")
```
 
---
 
#### `app.py`
 
`app.py` is responsible only for:
 
- FastAPI routes
- Request validation
- Response construction
- YOLO inference
- Business logic
Move all database configuration out of `app.py`. Remove all `sqlite3` imports and raw SQL from this file.
 
---
 
### 4. Replace Raw SQL with SQLAlchemy ORM
 
| Remove (raw SQLite) | Replace with (SQLAlchemy) |
|---|---|
| `sqlite3.connect()` | `db: Session = Depends(get_db)` |
| `cursor.execute("INSERT ...")` | `db.add(obj)` + `db.commit()` |
| `cursor.execute("SELECT ...")` | `db.query(Model).filter(...)` |
| `fetchone()` | `.first()` |
| `fetchall()` | `.all()` |
| `cursor.execute("DELETE ...")` | `db.delete(obj)` + `db.commit()` |
 
Always call `db.rollback()` on error.
 
---
 
### 5. Use FastAPI Dependency Injection
 
Every endpoint that requires database access must use:
 
```python
from sqlalchemy.orm import Session
from fastapi import Depends
from database import get_db
 
@app.post("/predict")
def predict(file: UploadFile, db: Session = Depends(get_db)):
    ...
```
 
Never manually create or close database connections inside route handlers.
 
---
 
### 6. Preserve Unrelated Functionality
 
Do not modify unless required by the database refactor:
 
- YOLO model loading
- Image validation
- Image saving
- Prediction rendering
- Metrics instrumentation
- Graceful shutdown
- Existing business logic
---
 
## Verification Checklist
 
Run the following before reporting the task complete:
 
```bash
cd services/yolo
pytest tests/
```
 
Verify each item:
 
- [ ] Application imports without errors
- [ ] All tests pass
- [ ] No `sqlite3` imports remain in the YOLO service
- [ ] No raw SQL strings remain
- [ ] SQLAlchemy ORM is used consistently throughout
- [ ] `Depends(get_db)` is used for all database sessions
- [ ] SQLite works with no environment variables set
- [ ] PostgreSQL works when `DB_BACKEND`, `DB_USER`, `DB_PASSWORD` are set
---
 
## Completion Criteria
 
Do not consider the task complete until **all** of the following are true:
 
- SQLAlchemy models are defined in `models.py`
- Database configuration exists in `database.py`
- FastAPI routes use `Depends(get_db)` wherever database access is needed
- Raw SQLite access has been fully removed
- Existing endpoint behavior is unchanged
- SQLite is the default backend
- PostgreSQL is supported via `DB_BACKEND`, `DB_USER`, `DB_PASSWORD`
- All tests pass
---
 
## Operational Rules
 
1. Preserve the existing API contract — no breaking changes.
2. Keep responsibilities separated: models in `models.py`, DB config in `database.py`, routes in `app.py`.
3. Do not mix `sqlite3` and SQLAlchemy.
4. Do not rewrite unrelated code.
5. Make the smallest implementation change required.
6. Never report success without running the verification step.