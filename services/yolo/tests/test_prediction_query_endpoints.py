from datetime import datetime

import database
from app import app, init_db
from fastapi.testclient import TestClient
from models import DetectionObject, PredictionSession


client = TestClient(app)


def setup_test_db(tmp_path):
    database.DB_PATH = str(tmp_path / "test_predictions.db")
    init_db()

    db = database.SessionLocal()
    try:
        sessions = [
            PredictionSession(
                uid="session-001",
                timestamp=datetime(2026, 6, 7, 10, 0, 0),
                original_image="uploads/original/img1.jpg",
                predicted_image="uploads/predicted/img1.jpg",
            ),
            PredictionSession(
                uid="session-002",
                timestamp=datetime(2026, 6, 7, 11, 0, 0),
                original_image="uploads/original/img2.jpg",
                predicted_image="uploads/predicted/img2.jpg",
            ),
            PredictionSession(
                uid="session-003",
                timestamp=datetime(2026, 6, 7, 12, 0, 0),
                original_image="uploads/original/img3.jpg",
                predicted_image="uploads/predicted/img3.jpg",
            ),
        ]
        db.add_all(sessions)

        detections = [
            DetectionObject(
                prediction_uid="session-001",
                label="person",
                score=0.96,
                box="[15, 25, 150, 250]",
            ),
            DetectionObject(
                prediction_uid="session-001",
                label="dog",
                score=0.82,
                box="[30, 40, 180, 280]",
            ),
            DetectionObject(
                prediction_uid="session-002",
                label="car",
                score=0.65,
                box="[100, 120, 250, 300]",
            ),
            DetectionObject(
                prediction_uid="session-002",
                label="person",
                score=0.51,
                box="[50, 60, 120, 220]",
            ),
            DetectionObject(
                prediction_uid="session-003",
                label="bicycle",
                score=0.33,
                box="[10, 15, 90, 140]",
            ),
        ]
        db.add_all(detections)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# -------------------------
# Label endpoint tests
# -------------------------
#since our DB is known, I want to check If the function for an example input , returns the ecpected output correctly
#in this example it was tested for label: person , and the output should be session1+2
def test_get_predictions_by_label_person(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/label/person")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 2

    returned_uids = [prediction["uid"] for prediction in data]

    assert "session-001" in returned_uids
    assert "session-002" in returned_uids

#check if the function identifies correctly labels that do not exsist in the DB
#INPUT: label : cat 
#EXPECTED OUTPUT: []
def test_get_predictions_by_label_no_matches(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/label/cat")

    assert response.status_code == 200
    assert response.json() == []

#check if label is cleaned properly after sending ( no label in a-b characters)
#INPUT: %20
#EXPECTED OUTPUT: 400 code , label cannot be empty
def test_get_predictions_by_label_empty(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/label/%20")

    assert response.status_code == 400
    assert response.json()["detail"] == "Label cannot be empty"
#check if label is cleaned properly after sending (label in a-b characters but has numeric chars)
#INPUT: %20person%20
#EXPECTED OUTPUT: 200 OK, 
def test_get_predictions_by_label_strips_whitespace(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/label/%20person%20")

    assert response.status_code == 200

    data = response.json()
    returned_uids = {prediction["uid"] for prediction in data}

    assert returned_uids == {"session-001", "session-002"}
# -------------------------
# Score endpoint tests
# -------------------------
#check if socre returns the filtered rows correctly
#INPUT: 0.8
#EXPECTED OUTPUT: person 0.96 and  dog 0.82
def test_get_detection_objects_by_score_returns_matching_objects(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/0.8")

    assert response.status_code == 200

    data = response.json()
    print(data)

    assert isinstance(data, list)
    assert len(data) == 2
    assert all(obj["score"] >= 0.8 for obj in data)

#check when entering a valid score , but none of the rows answer the condition, if it returns correctly an empty list
#INPUT: 1.0
#EXPECTED OUTPUT: []
def test_get_detection_objects_by_score_no_matches(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/1.0")

    assert response.status_code == 200
    assert response.json() == []

#check if the function handles correctly when sent an invalid input above threshold( score above 1 )
#INPUT: 1.5
#EXPECTED OUTPUT: 400 code , min_score must be between 0.0 and 1.0
def test_get_detection_objects_by_score_invalid_high(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/1.5")

    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"

#check if the function handles correctly when sent an invalid input below threshold (below 0 )
#INPUT: -0.1
#EXPECTED OUTPUT: 400 code , min_score must be between 0.0 and 1.0
def test_get_detection_objects_by_score_invalid_low(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/-0.1")

    assert response.status_code == 400
    assert response.json()["detail"] == "min_score must be between 0.0 and 1.0"


#check if the function handles correctly when sent score of 0.0
#INPUT: 0
#EXPECTED OUTPUT: 200 OK
def test_get_detection_objects_by_score_accepts_zero(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/0.0")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) == 5
    assert all(obj["score"] >= 0.0 for obj in data)


#check if the function handles correctly when sent score of 1.0
#INPUT: 1
#EXPECTED OUTPUT: 200 OK
def test_get_detection_objects_by_score_accepts_one(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/1.0")

    assert response.status_code == 200
    assert response.json() == []


#check if the function handles correctly when rows that answers the conditons of exactly equal
#INPUT: 0.33
#EXPECTED OUTPUT: 200 OK
def test_get_detection_objects_by_score_includes_equal_scores(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/predictions/score/0.33")

    assert response.status_code == 200

    data = response.json()

    assert any(obj["score"] == 0.33 for obj in data)
    assert all(obj["score"] >= 0.33 for obj in data)






    # to cover
def test_get_prediction_by_uid_success(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/prediction/session-001")

    assert response.status_code == 200

    data = response.json()

    assert data["uid"] == "session-001"
    assert data["timestamp"] == "2026-06-07 10:00:00"
    assert data["original_image"] == "uploads/original/img1.jpg"
    assert data["predicted_image"] == "uploads/predicted/img1.jpg"
    assert isinstance(data["detection_objects"], list)
    assert len(data["detection_objects"]) == 2


def test_get_prediction_by_uid_not_found(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/prediction/not-existing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"   

def test_get_prediction_image_not_found(tmp_path):
    setup_test_db(tmp_path)

    response = client.get("/prediction/session-001/image")

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"

def test_get_prediction_image_success(tmp_path):
    setup_test_db(tmp_path)

    predicted_file = tmp_path / "predicted.jpg"
    predicted_file.write_bytes(b"fake image content")

    db = database.SessionLocal()
    try:
        session = db.query(PredictionSession).filter(PredictionSession.uid == "session-001").first()
        session.predicted_image = str(predicted_file)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    response = client.get("/prediction/session-001/image")

    assert response.status_code == 200
    assert response.content == b"fake image content"
