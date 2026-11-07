from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features.online import (
    FEATURE_NAMES,
    build_online_features,
)
from src.api.main import create_app


DATA = Path("data/processed/uci")


def sample_statements():
    bills = [100, 200, -50, 300, 400, 500]
    payments = [10, 20, 30, 40, 50, 60]
    statuses = [-2, 0, 1, 2, 0, 3]

    return [
        {
            "statement_date": f"2026-{month:02d}-01",
            "repayment_status": statuses[i],
            "bill_amount": bills[i],
            "payment_amount": payments[i],
        }
        for i, month in enumerate(range(1, 7))
    ]


def test_online_feature_calculations():
    features = build_online_features(
        sample_statements(),
        credit_limit=5000,
    )

    assert set(features) == set(FEATURE_NAMES)
    assert features["months_late"] == 3
    assert features["latest_bill"] == 500
    assert features["bill_change"] == 400
    assert features["max_bill"] == 500
    assert features["total_payment"] == 210
    assert features["credit_utilization"] == pytest.approx(0.1)
    assert features["payment_ratio"] == pytest.approx(0.14)
    assert features["mean_bill"] == pytest.approx(1450 / 6)
    assert features["mean_repayment_status"] == pytest.approx(4 / 6)


def test_zero_positive_bill():
    statements = sample_statements()
    for row in statements:
        row["bill_amount"] = -10

    features = build_online_features(statements, 5000)

    assert np.isnan(features["payment_ratio"])
    assert features["credit_utilization"] == 0


@pytest.mark.skipif(
    not all(
        (DATA / name).exists()
        for name in (
            "history.parquet",
            "customers.parquet",
            "customer_features.parquet",
        )
    ),
    reason="Local UCI processed data unavailable",
)
def test_spark_online_feature_parity():
    history = pd.read_parquet(DATA / "history.parquet")
    customers = pd.read_parquet(DATA / "customers.parquet")
    spark_features = pd.read_parquet(
        DATA / "customer_features.parquet"
    )

    reference = spark_features.merge(
        customers[["customer_id", "credit_limit"]],
        on="customer_id",
        validate="one_to_one",
    )

    reference["credit_utilization"] = (
        reference["latest_bill"].clip(lower=0)
        / reference["credit_limit"].replace(0, np.nan)
    )

    reference["payment_ratio"] = (
        reference["total_payment"]
        / reference["total_positive_bill"].replace(0, np.nan)
    )

    reference = reference.set_index("customer_id")

    sample_ids = sorted(reference.index.tolist())[:25]
    history = history[history["customer_id"].isin(sample_ids)]

    checked = 0

    for customer_id, group in history.groupby("customer_id"):
        expected = reference.loc[customer_id]

        actual = build_online_features(
            group[
                [
                    "statement_date",
                    "repayment_status",
                    "bill_amount",
                    "payment_amount",
                ]
            ].to_dict("records"),
            credit_limit=expected["credit_limit"],
        )

        for name in FEATURE_NAMES:
            a = actual[name]
            b = expected[name]

            if pd.isna(a) and pd.isna(b):
                continue

            assert np.isclose(
                a, b, rtol=1e-9, atol=1e-6
            ), (
                f"Customer {customer_id}, feature {name}: "
                f"online={a}, spark={b}"
            )

        checked += 1

    assert checked == 25


def test_history_api_validation(tmp_path, monkeypatch):
    class FakeModel:
        feature_name_ = FEATURE_NAMES

        def predict_proba(self, frame, validate_features=False):
            assert validate_features
            assert list(frame.columns) == FEATURE_NAMES
            return np.array([[0.75, 0.25]])

    model_path = tmp_path / "fake.joblib"
    model_path.touch()

    monkeypatch.setattr(
        "src.api.main.joblib.load",
        lambda path: FakeModel(),
    )

    app = create_app(model_path=model_path)

    with TestClient(app) as client:
        payload = {
            "customer_id": "sample",
            "credit_limit": 5000,
            "statements": sample_statements(),
        }

        response = client.post(
            "/predict/history", json=payload
        )
        assert response.status_code == 200
        assert response.json()["default_probability"] == 0.25

        # Five statements instead of six.
        bad_payload = {
            **payload,
            "statements": payload["statements"][:5],
        }
        assert client.post(
            "/predict/history", json=bad_payload
        ).status_code == 422

        # Duplicate month.
        duplicate = sample_statements()
        duplicate[-1]["statement_date"] = duplicate[-2][
            "statement_date"
        ]
        assert client.post(
            "/predict/history",
            json={**payload, "statements": duplicate},
        ).status_code == 422

        # Negative credit limit.
        assert client.post(
            "/predict/history",
            json={**payload, "credit_limit": -1},
        ).status_code == 422
