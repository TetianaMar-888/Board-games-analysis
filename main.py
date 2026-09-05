"""
Board Game Rating API
Serves the Decision Tree classifier trained in Module 3 of the DM1 project.

The model predicts the community rating class (Low / Medium / High) of a board
game from its attributes. It was selected over KNN and Naive Bayes on macro-F1
and reached 0.700 macro-F1 on a held-out test set of 2,324 games.
"""

import json
import math
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path(__file__).parent

# ---------------------------------------------------------------- load artefacts
try:
    model = joblib.load(ARTIFACT_DIR / "model.pkl")
    with open(ARTIFACT_DIR / "model_meta.json") as f:
        meta = json.load(f)
except FileNotFoundError as exc:
    raise RuntimeError(
        f"Model artefacts not found in {ARTIFACT_DIR}. "
        "Copy model.pkl and model_meta.json into the app/ directory."
    ) from exc

FEATURES = meta["features"]
CLASSES = meta["classes"]
DEFAULTS = meta["defaults"]

app = FastAPI(
    title="Board Game Rating API",
    description=(
        "Predicts the BoardGameGeek community rating class of a board game. "
        "Model: Decision Tree, macro-F1 0.700 on held-out test data."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------- request schema
class GameFeatures(BaseModel):
    """Attributes of a board game. Counts are given in raw units; the API
    applies the log transforms the model expects."""

    com_weight: float = Field(
        ..., ge=1.0, le=5.3,
        description="Community complexity rating, 1-5 scale",
        examples=[2.8],
    )
    com_age_rec: float = Field(
        ..., ge=2, le=21,
        description="Community-recommended minimum age, in years",
        examples=[12],
    )
    year_published: int = Field(
        ..., ge=1900, le=2030,
        description="Year of publication",
        examples=[2018],
    )
    mfg_playtime: int = Field(
        ..., ge=1, le=60000,
        description="Manufacturer-stated playtime, in minutes",
        examples=[90],
    )
    num_want: int = Field(
        ..., ge=0,
        description="Number of users with the game on their wishlist",
        examples=[150],
    )
    num_user_ratings: int = Field(
        ..., ge=0,
        description="Number of user ratings the game has received",
        examples=[2500],
    )

    # Optional — fall back to the training-set defaults when omitted
    min_players: int | None = Field(None, ge=1, le=10, examples=[2])
    max_players: int | None = Field(None, ge=1, le=20, examples=[4])
    num_expansions: int | None = Field(None, ge=0, examples=[3])
    kickstarted: Literal[0, 1] | None = Field(None, examples=[0])
    is_reimplementation: Literal[0, 1] | None = Field(None, examples=[0])
    best_players_known: Literal[0, 1] | None = Field(None, examples=[1])
    cat_thematic: Literal[0, 1] | None = Field(None, examples=[0])
    cat_strategy: Literal[0, 1] | None = Field(None, examples=[1])
    cat_war: Literal[0, 1] | None = Field(None, examples=[0])
    cat_family: Literal[0, 1] | None = Field(None, examples=[0])
    cat_cgs: Literal[0, 1] | None = Field(None, examples=[0])
    cat_abstract: Literal[0, 1] | None = Field(None, examples=[0])
    cat_party: Literal[0, 1] | None = Field(None, examples=[0])
    cat_childrens: Literal[0, 1] | None = Field(None, examples=[0])


class Prediction(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    confidence: float


# ---------------------------------------------------------------- feature mapping
FIELD_TO_FEATURE = {
    "com_weight": "ComWeight",
    "com_age_rec": "ComAgeRec",
    "year_published": "YearPublished",
    "min_players": "MinPlayers",
    "max_players": "MaxPlayers",
    "num_expansions": "NumExpansions",
    "kickstarted": "Kickstarted",
    "is_reimplementation": "IsReimplementation",
    "best_players_known": "BestPlayers_known",
    "cat_thematic": "Cat:Thematic",
    "cat_strategy": "Cat:Strategy",
    "cat_war": "Cat:War",
    "cat_family": "Cat:Family",
    "cat_cgs": "Cat:CGS",
    "cat_abstract": "Cat:Abstract",
    "cat_party": "Cat:Party",
    "cat_childrens": "Cat:Childrens",
}


def build_feature_row(game: GameFeatures) -> pd.DataFrame:
    """Assemble a single-row DataFrame with the columns the model expects,
    in the order it was trained on."""
    row = dict(DEFAULTS)

    for field, feature in FIELD_TO_FEATURE.items():
        value = getattr(game, field)
        if value is not None:
            row[feature] = value

    # Log transforms — the model was trained on log1p of these counts
    row["MfgPlaytime_log"] = math.log1p(game.mfg_playtime)
    row["NumWant_log"] = math.log1p(game.num_want)
    row["NumUserRatings_log"] = math.log1p(game.num_user_ratings)

    missing = [f for f in FEATURES if f not in row]
    if missing:
        raise HTTPException(500, f"Internal error: missing features {missing}")

    return pd.DataFrame([row])[FEATURES]


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {
        "service": "Board Game Rating API",
        "model": meta["model_name"],
        "test_macro_f1": meta["test_macro_f1"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    """Model provenance and the defaults applied to omitted fields."""
    return {
        "model_name": meta["model_name"],
        "target": meta["target"],
        "classes": CLASSES,
        "n_features": len(FEATURES),
        "features": FEATURES,
        "test_macro_f1": meta["test_macro_f1"],
        "test_accuracy": meta["test_accuracy"],
        "sklearn_version": meta["sklearn_version"],
        "defaults_for_omitted_fields": DEFAULTS,
    }


@app.post("/predict", response_model=Prediction)
def predict(game: GameFeatures):
    """Predict the rating class of a single game."""
    X = build_feature_row(game)
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())

    return Prediction(
        predicted_class=CLASSES[idx],
        probabilities={cls: round(float(p), 4) for cls, p in zip(CLASSES, proba)},
        confidence=round(float(proba[idx]), 4),
    )


@app.post("/predict-batch")
def predict_batch(games: list[GameFeatures]):
    """Predict rating classes for up to 100 games in one request."""
    if not games:
        raise HTTPException(400, "Empty request body")
    if len(games) > 100:
        raise HTTPException(400, f"Batch limit is 100 games, received {len(games)}")

    X = pd.concat([build_feature_row(g) for g in games], ignore_index=True)
    probas = model.predict_proba(X)

    return {
        "count": len(games),
        "predictions": [
            {
                "predicted_class": CLASSES[int(p.argmax())],
                "probabilities": {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
                "confidence": round(float(p.max()), 4),
            }
            for p in probas
        ],
    }
