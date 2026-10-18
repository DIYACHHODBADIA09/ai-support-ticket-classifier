
from fastapi.testclient import TestClient

from main import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "AI Support Ticket Classifier API is running" in response.json()["message"]


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] is True


def test_predict_ticket():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "I cannot log in to my account"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["text"] == "I cannot log in to my account"
        assert data["predicted_category"] in {
            "technical",
            "billing",
            "account_access",
            "order_status",
        }
        assert 0 <= data["confidence"] <= 1
        assert "id" in data


def test_rejects_short_ticket():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "bad"},
        )

        assert response.status_code == 422


def test_list_tickets():
    with TestClient(app) as client:
        client.post(
            "/predict",
            json={"text": "Where is my package"},
        )

        response = client.get("/tickets")

        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert any(
            ticket["text"] == "Where is my package"
            for ticket in response.json()
        )
