import os
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("CONFIDENCE_THRESHOLD", "0.5")

from app import app, init_db
import database



TEST_IMAGE = os.path.join(os.path.dirname(__file__), "data", "beatles.jpeg")


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_predictions.db")
    monkeypatch.setattr(database, "DB_PATH", db_file)
    init_db()


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

 #check for file type , must be an image
def test_predict_rejects_unsupported_file_type(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.pdf",
                b"this is not an image",
                "application/pdf"
            )
        }
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "File type not supported"}


