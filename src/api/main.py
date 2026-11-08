"""Credit-risk inference API with model versioning and safe logging."""

import hashlib
import json
import logging
import math
import os
import time
import uuid

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

from src.features.online import (
    FEATURE_NAMES,
    build_online_features,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DEFAULT_MODEL_PATH = Path(
    "outputs/uci/lightgbm_fair/model.joblib"
)

MODEL_NAME = "lightgbm_final_12_features"
Numeric = StrictInt | StrictFloat

logger = logging.getLogger("credit_risk")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

logger.propagate = False


def log_event(event, **fields):
    """Write JSON logs without customer inputs or identifiers."""
    record = {"event": event, **fields}
    logger.info(json.dumps(record, allow_nan=False))


# --------------------------------------------------
# 2. Request and response schemas
# --------------------------------------------------

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

        if any(
            b - a != 1
            for a, b in zip(months, months[1:])
        ):
            raise ValueError(
                "Statements must cover six distinct, "
                "consecutive months."
            )

        return self


class PredictionResponse(BaseModel):
    customer_id: str | int | None
    default_probability: float
    model_name: str
    model_version: str
    prediction_id: str


# --------------------------------------------------
# 3. Application
# --------------------------------------------------

def create_app(model_path=None):
    path = Path(
        model_path
        or os.environ.get(
            "CREDIT_MODEL_PATH",
            DEFAULT_MODEL_PATH,
        )
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}"
            )

        model = joblib.load(path)

        names = getattr(model, "feature_name_", None)

        if names is None:
            names = getattr(
                model, "feature_names_in_", None
            )

        if names is None:
            raise ValueError(
                "Model does not contain feature names."
            )

        names = list(names)

        if not names or len(set(names)) != len(names):
            raise ValueError("Invalid feature schema.")

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Model does not support predict_proba."
            )

        # Model artifact hash identifies the exact file loaded.
        model_version = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()[:12]

        app.state.model = model
        app.state.feature_names = names
        app.state.model_version = model_version

        log_event(
            "model_loaded",
            model_name=MODEL_NAME,
            model_version=model_version,
            n_features=len(names),
        )

        yield

        app.state.model = None

    app = FastAPI(
        title="Credit Sequence Risk API",
        description="Versioned credit-default inference",
        version="0.3.0",
        lifespan=lifespan,
    )

    def make_prediction(
        customer_id,
        provided,
        request: Request,
    ):
        prediction_id = uuid.uuid4().hex
        started = time.perf_counter()

        expected = request.app.state.feature_names

        missing = sorted(set(expected) - set(provided))
        extra = sorted(set(provided) - set(expected))

        if missing or extra:
            log_event(
                "prediction_rejected",
                prediction_id=prediction_id,
                reason="feature_schema",
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra,
                },
                headers={"X-Prediction-ID": prediction_id},
            )

        for name, value in provided.items():
            if value is not None and not math.isfinite(value):
                log_event(
                    "prediction_rejected",
                    prediction_id=prediction_id,
                    reason="non_finite_input",
                )
                raise HTTPException(
                    status_code=422,
                    detail=f"Non-finite value for {name}",
                    headers={"X-Prediction-ID": prediction_id},
                )

        row = {
            name: (
                np.nan
                if provided[name] is None
                else float(provided[name])
            )
            for name in expected
        }

        frame = pd.DataFrame(
            [row], columns=expected
        )

        try:
            probabilities = (
                request.app.state.model.predict_proba(
                    frame,
                    validate_features=True,
                )
            )
            probability = float(probabilities[0, 1])

            if not math.isfinite(probability):
                raise ValueError(
                    "Non-finite model prediction."
                )

            if not 0 <= probability <= 1:
                raise ValueError(
                    "Prediction outside [0, 1]."
                )

        except Exception:
            # Log the failure without raw features or customer IDs.
            log_event(
                "prediction_failed",
                prediction_id=prediction_id,
                model_version=request.app.state.model_version,
            )

            raise HTTPException(
                status_code=500,
                detail="Model inference failed.",
                headers={"X-Prediction-ID": prediction_id},
            )

        duration_ms = round(
            (time.perf_counter() - started) * 1000,
            3,
        )

        log_event(
            "prediction_succeeded",
            prediction_id=prediction_id,
            model_name=MODEL_NAME,
            model_version=request.app.state.model_version,
            latency_ms=duration_ms,
        )

        return PredictionResponse(
            customer_id=customer_id,
            default_probability=probability,
            model_name=MODEL_NAME,
            model_version=request.app.state.model_version,
            prediction_id=prediction_id,
        )

    # --------------------------------------------------
    # 4. Endpoints
    # --------------------------------------------------

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ok",
            "model_loaded": (
                request.app.state.model is not None
            ),
            "n_features": len(
                request.app.state.feature_names
            ),
            "model_version": (
                request.app.state.model_version
            ),
        }

    @app.get("/features")
    def features(request: Request):
        return {
            "required_features": (
                request.app.state.feature_names
            ),
        }

    @app.post(
        "/predict",
        response_model=PredictionResponse,
    )
    def predict(
        payload: PredictionRequest,
        request: Request,
    ):
        return make_prediction(
            payload.customer_id,
            payload.features,
            request,
        )

    @app.post(
        "/predict/history",
        response_model=PredictionResponse,
    )
    def predict_history(
        payload: HistoryPredictionRequest,
        request: Request,
    ):
        if set(
            request.app.state.feature_names
        ) != set(FEATURE_NAMES):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model schema incompatible "
                    "with raw-history inference."
                ),
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
