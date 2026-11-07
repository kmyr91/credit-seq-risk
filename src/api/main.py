"""FastAPI service for engineered-feature and raw-history predictions."""

import math
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    model_validator,
)

from src.features.online import FEATURE_NAMES, build_online_features


DEFAULT_MODEL_PATH = Path(
    "outputs/uci/stacking/lightgbm_full_control_model.joblib"
)

Numeric = StrictInt | StrictFloat


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | int | None = None
    features: dict[str, Numeric | None]


class MonthlyStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement_date: date
    repayment_status: StrictInt
    bill_amount: Numeric
    payment_amount: Numeric

    @model_validator(mode="after")
    def check_values(self):
        for name in ("bill_amount", "payment_amount"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")

        if self.statement_date.day != 1:
            raise ValueError(
                "statement_date must be the first day of the month."
            )
        return self


class HistoryPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | int | None = None
    credit_limit: Numeric
    statements: list[MonthlyStatement] = Field(
        min_length=6,
        max_length=6,
    )

    @model_validator(mode="after")
    def validate_history(self):
        if not math.isfinite(self.credit_limit):
            raise ValueError("credit_limit must be finite.")

        if self.credit_limit <= 0:
            raise ValueError("credit_limit must be positive.")

        months = sorted(
            s.statement_date.year * 12 + s.statement_date.month
            for s in self.statements
        )

        if any(b - a != 1 for a, b in zip(months, months[1:])):
            raise ValueError(
                "Statements must cover six distinct, "
                "consecutive months."
            )

        return self


class PredictionResponse(BaseModel):
    customer_id: str | int | None
    default_probability: float
    model_name: str


def create_app(model_path=None):
    path = Path(
        model_path
        or os.environ.get("CREDIT_MODEL_PATH", DEFAULT_MODEL_PATH)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path.resolve()}")

        model = joblib.load(path)

        feature_names = getattr(model, "feature_name_", None)
        if feature_names is None:
            feature_names = getattr(model, "feature_names_in_", None)

        if feature_names is None:
            raise ValueError("Saved model has no feature names.")

        feature_names = list(feature_names)

        if not feature_names or len(set(feature_names)) != len(
            feature_names
        ):
            raise ValueError("Invalid model feature schema.")

        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support predict_proba.")

        app.state.model = model
        app.state.feature_names = feature_names

        print(
            f"Loaded model: {path} "
            f"({len(feature_names)} features)",
            flush=True,
        )

        yield

        app.state.model = None

    app = FastAPI(
        title="Credit Sequence Risk API",
        description="Credit default probability prediction",
        version="0.2.0",
        lifespan=lifespan,
    )

    def make_prediction(
        customer_id, provided, request: Request
    ):
        expected = request.app.state.feature_names

        missing = sorted(set(expected) - set(provided))
        extra = sorted(set(provided) - set(expected))

        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra,
                },
            )

        for name, value in provided.items():
            if value is not None and not math.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Non-finite value for {name}",
                )

        row = {
            name: (
                np.nan if provided[name] is None
                else float(provided[name])
            )
            for name in expected
        }

        frame = pd.DataFrame([row], columns=expected)

        probability = float(
            request.app.state.model.predict_proba(
                frame, validate_features=True
            )[0, 1]
        )

        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid probability.",
            )

        return PredictionResponse(
            customer_id=customer_id,
            default_probability=probability,
            model_name="lightgbm_full_control",
        )

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ok",
            "model_loaded": request.app.state.model is not None,
            "n_features": len(request.app.state.feature_names),
        }

    @app.get("/features")
    def features(request: Request):
        return {
            "required_features": request.app.state.feature_names,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        return make_prediction(
            payload.customer_id,
            payload.features,
            request,
        )

    @app.post("/predict/history", response_model=PredictionResponse)
    def predict_history(
        payload: HistoryPredictionRequest,
        request: Request,
    ):
        if set(request.app.state.feature_names) != set(FEATURE_NAMES):
            raise HTTPException(
                status_code=503,
                detail="Model schema incompatible with raw-history inference.",
            )

        engineered = build_online_features(
            payload.statements,
            payload.credit_limit,
        )

        return make_prediction(
            payload.customer_id,
            engineered,
            request,
        )

    return app


app = create_app()
