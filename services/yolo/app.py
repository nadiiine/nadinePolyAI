from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy.orm import Session
from ultralytics import YOLO
from PIL import Image
import logging
import os
import uuid
import shutil
import time
import signal
import sys
import boto3
from pydantic import BaseModel

from database import get_db, init_database
from models import DetectionObject, PredictionSession

is_shutting_down = False

def handle_sigterm(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.info("Received SIGTERM. Shutting down gracefully...")
    logging.info("Cleanup done. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

# Expose /metrics endpoint with default process metrics + FastAPI HTTP metrics
Instrumentator().instrument(app).expose(app)

# Confidence threshold for object detection (0.0 - 1.0).
# Detections below this score are discarded.
# Override with: export CONFIDENCE_THRESHOLD=0.7
_raw_threshold = os.environ.get("CONFIDENCE_THRESHOLD")
if _raw_threshold is not None:
    CONFIDENCE_THRESHOLD = float(_raw_threshold)
    logging.info(f"CONFIDENCE_THRESHOLD set to {CONFIDENCE_THRESHOLD} (from environment)")
else:  # pragma: no cover
    CONFIDENCE_THRESHOLD = 0.5
    logging.info(f"CONFIDENCE_THRESHOLD not set, using default: {CONFIDENCE_THRESHOLD}")

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")
AWS_REGION = os.environ["AWS_REGION"]
AWS_S3_BUCKET = os.environ["AWS_S3_BUCKET"]

s3 = boto3.client("s3", region_name=AWS_REGION)


class PredictRequest(BaseModel):
    image_s3_key: str
    chat_id: str
    prediction_id: str
    image_name: str

def _format_timestamp(timestamp):
    if hasattr(timestamp, "strftime"):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp


@app.post("/predict")
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Predict objects in an image
    """
    start_time = time.time()

    ext = os.path.splitext(request.image_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="File type not supported"
        )

    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    s3.download_file(
        AWS_S3_BUCKET,
        request.image_s3_key,
        original_path,
    )

    results = model(original_path, device="cpu", conf=CONFIDENCE_THRESHOLD)

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    predicted_s3_key = (
        f"{request.chat_id}/{request.prediction_id}/predicted/{request.image_name}"
    )

    s3.upload_file(
        predicted_path,
        AWS_S3_BUCKET,
        predicted_s3_key,
        ExtraArgs={"ContentType": "image/jpeg"},
    )

    session = PredictionSession(
        uid=uid,
        original_image=request.image_s3_key,
        predicted_image=predicted_s3_key,
    )
    db.add(session)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        db.add(DetectionObject(
            prediction_uid=uid,
            label=label,
            score=score,
            box=str(bbox),
        ))
        detected_labels.append(label)

    db.commit()

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time,
        "original_image_s3_key": request.image_s3_key,
        "predicted_image_s3_key": predicted_s3_key,
    }

@app.get("/prediction/{uid}")
def get_prediction_by_uid(uid: str, db: Session = Depends(get_db)):
    """
    Get prediction session by uid with all detected objects
    """
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")

    objects = db.query(DetectionObject).filter_by(prediction_uid=uid).all()

    return {
        "uid": session.uid,
        "timestamp": _format_timestamp(session.timestamp),
        "original_image": session.original_image,
        "predicted_image": session.predicted_image,
        "detection_objects": [
            {
                "id": obj.id,
                "label": obj.label,
                "score": obj.score,
                "box": obj.box
            } for obj in objects
        ]
    }


@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, db: Session = Depends(get_db)):
    """
    Return the annotated (bounding-box) image for a prediction
    """
    session = db.query(PredictionSession).filter_by(uid=uid).first()
    if not session or not os.path.exists(session.predicted_image):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(session.predicted_image)

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db)):  # receives a label string from the URL path parameter
    """
    Return all prediction sessions that contain at least one detected object
    with the given label.
    """
    
    label = label.strip()  # remove leading/trailing spaces from the input

    if not label:
        raise HTTPException(
            status_code=400,
            detail="Label cannot be empty"
        )  # return error if the label is empty

    rows = (
        db.query(PredictionSession, DetectionObject)
        .join(DetectionObject, PredictionSession.uid == DetectionObject.prediction_uid)
        .filter(DetectionObject.label == label)
        .all()
    )

    predictions = {}  # dictionary used to group objects under the same prediction session

    for session, detection in rows:  # iterate over every row returned from the query

        uid = session.uid  # get prediction session uid

        if uid not in predictions:
            # first time we see this prediction session
            predictions[uid] = {
                "uid": session.uid,
                "timestamp": _format_timestamp(session.timestamp),
                "detection_objects": []
            }

        predictions[uid]["detection_objects"].append({
            "id": detection.id,
            "label": detection.label,
            "score": detection.score,
            "box": detection.box
        })  # add current detection object to its prediction session

    return list(predictions.values())  # return grouped prediction sessions as a list



#2
@app.get("/predictions/score/{min_score}")
def get_detection_objects_by_score(min_score: float, db: Session = Depends(get_db)):  
    # receives min_score from the URL path parameter as a float

    """
    Return all detection objects with confidence score >= min_score.
    """

    if min_score < 0.0 or min_score > 1.0:
        # confidence score must be between 0.0 and 1.0
        raise HTTPException(
            status_code=400,
            detail="min_score must be between 0.0 and 1.0"
        )

    objects = (
        db.query(DetectionObject)
        .filter(DetectionObject.score >= min_score)
        .all()
    )

    result = []

    for obj in objects:
     result.append({
        "id": obj.id,
        "prediction_uid": obj.prediction_uid,
        "label": obj.label,
        "score": obj.score,
        "box": obj.box
    })



    return result

    #service ready check
@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}
    # return all matching detection objects as a list
@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__": # pragma: no cover
    import uvicorn

    init_database()
    
    uvicorn.run(app, host="0.0.0.0", port=8081)
