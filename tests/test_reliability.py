import json

import numpy as np
import pytest

from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeModel:
    feature_name_ = ["mean_bill", "credit_limit"]

    def predict_proba(self, frame, validate_features=False):
        assert validate_features
        return np.array([[0.8, 0.2]])


class BrokenModel(FakeModel):
    def predict_proba(self, frame, validate_features=False):
        raise RuntimeError("Internal model error: secret detail")


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    def factory(model):
        path = tmp_path / "model.joblib"
        path.write_bytes(b"test-artifact-v1")

        monkeypatch.setattr(
            "src.api.main.joblib.load",
            lambda _: model,
        )

        return TestClient(create_app(model_path=path))

    return factory


def payload():
    return {
        "customer_id": "test-001",
        "features": {
            "mean_bill": 12000,
            "credit_limit": 50000,
        },
    }


def test_model_version_in_health_and_prediction(make_client):
    with make_client(FakeModel()) as client:
        health = client.get("/health").json()
        result = client.post("/predict", json=payload()).json()

        assert len(health["model_version"]) == 12
        assert result["model_version"] == health["model_version"]
        assert result["model_name"] == "lightgbm_full_control"


def test_unique_prediction_ids(make_client):
    with make_client(FakeModel()) as client:
        first = client.post("/predict", json=payload()).json()
        second = client.post("/predict", json=payload()).json()

        assert first["prediction_id"]
        assert second["prediction_id"]
        assert first["prediction_id"] != second["prediction_id"]


def test_model_failure_is_handled(make_client):
    with make_client(BrokenModel()) as client:
        response = client.post("/predict", json=payload())

        assert response.status_code == 500
        assert response.json()["detail"] == "Model inference failed."
        assert "secret detail" not in response.text
        assert response.headers.get("X-Prediction-ID")


def test_model_version_changes_with_artifact(
    tmp_path, monkeypatch
):
    path = tmp_path / "model.joblib"
    monkeypatch.setattr(
        "src.api.main.joblib.load",
        lambda _: FakeModel(),
    )

    path.write_bytes(b"version-one")

    with TestClient(create_app(model_path=path)) as client:
        version_one = client.get("/health").json()["model_version"]

    path.write_bytes(b"version-two")

    with TestClient(create_app(model_path=path)) as client:
        version_two = client.get("/health").json()["model_version"]

    assert version_one != version_two
