"""Training pipeline with LightGBM, XGBoost and ensemble blending."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb

from src.evaluate import print_scores, zindi_score

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Default hyperparameters (tuned baseline)
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1086 / 735,  # neg/pos ratio from training data
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

N_SPLITS = 5
EARLY_STOPPING = 100


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lgb",
    n_splits: int = N_SPLITS,
    params: dict | None = None,
) -> tuple[np.ndarray, list]:
    """Stratified K-Fold cross-validation.

    Returns:
        oof_proba: out-of-fold probabilities (n_samples,)
        models: list of trained model objects
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = _fit(X_tr, y_tr, X_val, y_val, model_type, params)
        oof[val_idx] = _predict_proba(model, X_val, model_type)
        score = zindi_score(y_val.values, (oof[val_idx] >= 0.5).astype(int), oof[val_idx])
        print(f"  Fold {fold}/{n_splits}  Zindi={score:.4f}")
        models.append(model)

    print("\nOOF overall:")
    print_scores(y.values, oof)
    return oof, models


def _fit(X_tr, y_tr, X_val, y_val, model_type: str, params: dict | None):
    p = (params or {})
    if model_type == "lgb":
        p = {**LGB_PARAMS, **p}
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
    elif model_type == "xgb":
        p = {**XGB_PARAMS, **p}
        model = xgb.XGBClassifier(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def _predict_proba(model, X: pd.DataFrame, model_type: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def ensemble_predict(models: list, X: pd.DataFrame, model_type: str) -> np.ndarray:
    """Average predictions from all fold models."""
    preds = np.stack([_predict_proba(m, X, model_type) for m in models], axis=1)
    return preds.mean(axis=1)


def blend(probas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted blend of multiple probability arrays."""
    if weights is None:
        weights = [1.0 / len(probas)] * len(probas)
    w = np.array(weights) / sum(weights)
    return sum(p * wi for p, wi in zip(probas, w))


def save_models(models: list, model_type: str, path: Path | None = None) -> Path:
    path = path or (MODEL_DIR / f"{model_type}_models.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"Saved {len(models)} models → {path}")
    return path


def load_models(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from src.features import prepare_Xy

    print("Loading and engineering features …")
    X_train, y_train, X_test = prepare_Xy(
        str(DATA_DIR / "Train.csv"),
        str(DATA_DIR / "Test.csv"),
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n=== LightGBM ===")
    lgb_oof, lgb_models = cross_validate(X_train, y_train, model_type="lgb")

    print("\n=== XGBoost ===")
    xgb_oof, xgb_models = cross_validate(X_train, y_train, model_type="xgb")

    # blend
    blended_oof = blend([lgb_oof, xgb_oof], weights=[0.6, 0.4])
    print("\nBlended OOF:")
    print_scores(y_train.values, blended_oof)

    save_models(lgb_models, "lgb")
    save_models(xgb_models, "xgb")
    print("Done.")
