# Board Game Rating API

Serves the Decision Tree classifier trained in Module 3 of the DM1 project. Given
a board game's attributes, it predicts the BoardGameGeek community rating class:
**Low**, **Medium** or **High**.

Model performance on a held-out test set of 2,324 games: macro-F1 **0.700**,
accuracy **0.699**, macro ROC-AUC **0.855** (majority-class baseline: 0.463
accuracy, 0.211 macro-F1).

---

## Setup

The model artefacts are not in this repository. Copy them from the training
notebook output into `app/`:

```
app/model.pkl          # the fitted sklearn Pipeline
app/model_meta.json    # feature list, class order, defaults, metrics
```

Both are written by the final cell of `notebooks/DM1_03_classification_regression.ipynb`.

### Run with Docker

```bash
docker build -t bgg-rating-api .
docker run -p 8000:8000 bgg-rating-api
```

### Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive documentation: http://localhost:8000/docs

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service name and headline metric |
| GET | `/health` | Liveness check |
| GET | `/model-info` | Feature list, class order, metrics, defaults |
| POST | `/predict` | Classify one game |
| POST | `/predict-batch` | Classify up to 100 games |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "com_weight": 3.5,
    "com_age_rec": 14,
    "year_published": 2019,
    "mfg_playtime": 120,
    "num_want": 800,
    "num_user_ratings": 15000
  }'
```

```json
{
  "predicted_class": "High",
  "probabilities": {"Low": 0.02, "Medium": 0.19, "High": 0.79},
  "confidence": 0.79
}
```

---

## Input

Six fields are required. They are the six most important features in the trained
model, together accounting for 92% of its total feature importance:

| Field | Unit | Range |
|---|---|---|
| `com_weight` | complexity, 1–5 scale | 1.0 – 5.3 |
| `com_age_rec` | years | 2 – 21 |
| `year_published` | year | 1900 – 2030 |
| `mfg_playtime` | minutes | 1 – 60000 |
| `num_want` | users with the game wishlisted | ≥ 0 |
| `num_user_ratings` | ratings received | ≥ 0 |

Fourteen further fields are optional — player counts, expansion count, three
binary flags and eight category flags. Omitted fields fall back to the defaults
listed by `/model-info` (training-set medians for player counts, zero for flags).

**Counts are sent in raw units.** The model was trained on `log1p` of playtime,
wishlist count and rating count; the API applies the transform, so callers never
handle logarithms.

---

## Design notes

**The artefact is a `Pipeline`, not a bare estimator.** Any preprocessing the
model needs travels with it, so the API cannot drift out of step with training —
a class of bug that is otherwise easy to introduce and hard to detect.

**The scikit-learn version is pinned exactly** in `requirements.txt`. Unpickling
a model under a different minor version either raises an error or, worse, loads
successfully and behaves differently.

**Field names are decoupled from feature names.** The API accepts `snake_case`
input and maps it to the training column names (`Cat:War`, `NumWant_log`) in one
place, so renaming a column in the notebook does not break the public contract.

---

## Limitations

The model was trained on 15,487 games that carry community assessments of
complexity, recommended age and playtime. Games without those assessments — older
and rarely-voted-on titles — were excluded, and predictions for such games are
outside the tested range.

`NumWant` is the strongest single predictor (feature importance 0.413), and it
accumulates over a game's lifetime. Predictions for newly published games, which
have little wishlist history, are correspondingly less reliable than the test
figures suggest.

The target and the popularity features all originate from the same BoardGameGeek
community, so the model describes the internal consistency of one community's
judgements rather than any external measure of quality.
