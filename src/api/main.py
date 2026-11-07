import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt


DEFAULT_MODEL_PATH = Path(
    "outputs/uci/stacking/lightgbm_full_control_model.joblib"
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str | int | None = None
    features: dict[str, StrictFloat | StrictInt | None]


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
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}"
            )

        model = joblib.load(path)

        feature_names = getattr(
            model, "feature_name_", None
        )

        if feature_names is None:
            feature_names = getattr(
                model, "feature_names_in_", None
            )

        if feature_names is None:
            raise ValueError(
                "Saved model has no feature names."
            )

        feature_names = list(feature_names)

        if not feature_names or len(set(feature_names)) != len(
            feature_names
        ):
            raise ValueError("Invalid model feature schema.")

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Model does not support predict_proba."
            )

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
        description="Credit-default probability inference",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "ok",
            "model_loaded": request.app.state.model is not None,
            "n_features": len(
                request.app.state.feature_names
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
    def predict(payload: PredictionRequest, request: Request):

        expected = request.app.state.feature_names
        provided = payload.features

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

        # Reorder by the exact feature schema of the
        # trained LightGBM model. None becomes NaN.
        row = {
            name: (
                np.nan
                if provided[name] is None
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

        if not math.isfinite(probability) or not (
            0 <= probability <= 1
        ):
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid probability.",
            )

        return PredictionResponse(
            customer_id=payload.customer_id,
            default_probability=probability,
            model_name="lightgbm_full_control",
        )

    return app


app = create_app()
