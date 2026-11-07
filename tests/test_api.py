from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeModel:
    feature_name_ = ["mean_bill", "credit_limit"]

    def predict_proba(self, frame, validate_features=False):
        assert validate_features
        assert list(frame.columns) == self.feature_name_
        assert frame.shape == (1, 2)
        return np.array([[0.8, 0.2]])


@pytest.fixture
def client(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.touch()

    monkeypatch.setattr(
        "src.api.main.joblib.load",
        lambda path: FakeModel(),
    )

    app = create_app(model_path=model_path)

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["n_features"] == 2


def test_feature_schema(client):
    response = client.get("/features")

    assert response.status_code == 200
    assert response.json()["required_features"] == [
        "mean_bill",
        "credit_limit",
    ]


def test_predict_success(client):
    response = client.post(
        "/predict",
        json={
            "customer_id": "test-001",
            "features": {
                "credit_limit": 50000,
                "mean_bill": 12000,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["customer_id"] == "test-001"
    assert response.json()["default_probability"] == 0.2


def test_missing_feature(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "credit_limit": 50000
            }
        },
    )

    assert response.status_code == 422
    assert "mean_bill" in response.json()["detail"][
        "missing_features"
    ]


def test_extra_feature(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean_bill": 12000,
                "credit_limit": 50000,
                "unknown": 1,
            }
        },
    )

    assert response.status_code == 422


def test_invalid_type(client):
    response = client.post(
        "/predict",
        json={
            "features": {
                "mean_bill": "not a number",
                "credit_limit": 50000,
            }
        },
    )

    assert response.status_code == 422


def test_missing_model_fails_at_startup(tmp_path):
    app = create_app(
        model_path=tmp_path / "missing.joblib"
    )

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass
