"""Generate submission file from trained models."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR  = Path(__file__).parent.parent / "models"
SUBMIT_DIR = Path(__file__).parent.parent / "data" / "submissions"


def load_models(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return preds.mean(axis=1)


def make_submission(
    test_ids: pd.Index,
    proba: np.ndarray,
    threshold: float = 0.5,
    tag: str = "submission",
) -> pd.DataFrame:
    sub = pd.DataFrame({
        "ID": test_ids,
        "TargetF1": (proba >= threshold).astype(int),
        "TargetRAUC": proba,
    })
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMIT_DIR / f"{tag}.csv"
    sub.to_csv(path, index=False)
    print(f"Submission saved → {path}  ({sub['TargetF1'].sum()} positives / {len(sub)} total)")
    return sub


def load_threshold(key: str = "blend", default: float = 0.5) -> float:
    """Load the OOF-optimised threshold saved by train.py."""
    path = MODEL_DIR / "thresholds.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        thr = float(data.get(key, default))
        print(f"Loaded threshold [{key}] = {thr:.4f}")
        return thr
    print(f"No thresholds.json found, using default={default}")
    return default


if __name__ == "__main__":
    from src.features import prepare_Xy
    from src.train import blend

    print("Engineering features …")
    _, _, X_test = prepare_Xy(
        str(DATA_DIR / "Train.csv"),
        str(DATA_DIR / "Test.csv"),
    )

    lgb_models = load_models(MODEL_DIR / "lgb_models.pkl")
    xgb_models = load_models(MODEL_DIR / "xgb_models.pkl")

    lgb_proba = predict_proba(lgb_models, X_test)
    xgb_proba = predict_proba(xgb_models, X_test)
    blended   = blend([lgb_proba, xgb_proba], weights=[0.6, 0.4])

    threshold = load_threshold("blend")
    make_submission(X_test.index, blended, threshold=threshold, tag="lgb_xgb_blend_v2")
    print("Done.")
