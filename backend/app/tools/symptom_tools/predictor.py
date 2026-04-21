"""
XGBoost symptom predictor — model loading and raw inference.

This module owns:
  - loading the model and metadata once at import time
  - encoding categorical fields via stored LabelEncoders
  - running predict_proba and returning a structured result

It does NOT make decisions about confidence or uncertainty —
that belongs in confidence.py and Agent 2.
"""
import pickle
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH = "artifacts/xgb_vet_verified_3000_v3.json"
META_PATH  = "artifacts/xgb_vet_verified_3000_v3_meta.pkl"


# ── Internal model state ───────────────────────────────────────────────────────

_xgb_model:       Optional[XGBClassifier] = None
_feature_columns: Optional[List[str]]     = None
_label_encoders:  Optional[Dict]          = None
_diag_classes:    Optional[List[str]]     = None
_load_error:      Optional[str]           = None


def _load() -> None:
    """Load model + metadata into module-level globals. Raises on failure."""
    global _xgb_model, _feature_columns, _label_encoders, _diag_classes

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    _xgb_model       = model
    _feature_columns = meta["feature_columns"]
    _label_encoders  = meta["label_encoders"]
    _diag_classes    = meta["diag_classes"]


def _ensure_loaded() -> Optional[str]:
    """
    Ensure the model is loaded. Returns an error string if it fails, else None.
    Retries lazily if the initial startup load failed.
    """
    global _load_error
    if _xgb_model is not None:
        return None
    try:
        _load()
        _load_error = None
        return None
    except Exception:
        _load_error = traceback.format_exc()
        return _load_error


# Try once at import time so the first request isn't slow
_ensure_loaded()


# ── Encoding ───────────────────────────────────────────────────────────────────

def _encode(raw: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply stored LabelEncoders to categorical columns and return a
    correctly-ordered DataFrame ready for predict_proba.

    Unknown label values fall back to 0 (matching training-time behaviour).
    """
    d = raw.copy()
    for col, le in _label_encoders.items():
        if col in d:
            try:
                d[col] = int(le.transform([d[col]])[0])
            except ValueError:
                d[col] = 0
    return pd.DataFrame([d])[_feature_columns]


# ── Public inference function ──────────────────────────────────────────────────

class PredictionResult:
    """Thin dataclass-style holder for raw model output."""

    def __init__(
        self,
        top_label: str,
        top_confidence: float,
        top3: List[Dict[str, Any]],
        probability_map: Dict[str, float],
    ) -> None:
        self.top_label       = top_label
        self.top_confidence  = top_confidence
        self.top3            = top3          # includes the winner at index 0
        self.probability_map = probability_map


def predict(raw: Dict[str, Any]) -> PredictionResult:
    """
    Run XGBoost inference on a structured pet case dict.

    Args:
        raw: Dict with pet profile + binary symptom flags.
             Required: species, breed, sex, neutered, age_years, weight_kg.
             Optional symptom flags default to 0 if missing.

    Returns:
        PredictionResult with raw model outputs.

    Raises:
        RuntimeError: If the model could not be loaded.
    """
    err = _ensure_loaded()
    if err:
        raise RuntimeError(f"Symptom model failed to load:\n{err}")

    df      = _encode(raw)
    probs   = _xgb_model.predict_proba(df)[0]
    sorted_ = np.argsort(probs)[::-1]

    top3 = [
        {"condition": _diag_classes[int(i)], "confidence": round(float(probs[int(i)]), 4)}
        for i in sorted_[:3]
    ]
    prob_map = {
        _diag_classes[i]: round(float(probs[i]), 4)
        for i in range(len(_diag_classes))
    }

    return PredictionResult(
        top_label      = _diag_classes[int(sorted_[0])],
        top_confidence = round(float(probs[sorted_[0]]), 4),
        top3           = top3,
        probability_map= prob_map,
    )
