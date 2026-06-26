from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from ultralytics import YOLO
from PIL import Image
import logging
import os
import uuid
import time
import signal
import sys
import boto3
from pydantic import BaseModel

is_shutting_down = False

def handle_sigterm(signum, frame):
    global is_shutting_down
    is_shutting_down = True
    logging.info("Received SIGTERM. Shutting down gracefully...")
    logging.info("Cleanup done. Exiting.")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

Instrumentator().instrument(app).expose(app)

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

model = YOLO("yolov8n.pt")

AWS_REGION = os.environ["AWS_REGION"]
AWS_S3_BUCKET = os.environ["AWS_S3_BUCKET"]

s3 = boto3.client("s3", region_name=AWS_REGION)


class PredictRequest(BaseModel):
    image_s3_key: str
    chat_id: str
    prediction_id: str
    image_name: str


@app.post("/predict")
def predict(request: PredictRequest):
    start_time = time.time()

    ext = os.path.splitext(request.image_name)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported")

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

    detected_labels = []

    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)

    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time,
        "original_image_s3_key": request.image_s3_key,
        "predicted_image_s3_key": predicted_s3_key,
    }


@app.get("/ready")
def ready():
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Service is shutting down")
    return {"status": "ready"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)