"""Production-style checks for a deployed frequency model: input drift (PSI) and actual-vs-expected claims."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_poisson_deviance

from claims_model import models

# Common industry thresholds for the population stability index.
PSI_WATCH, PSI_ALERT = 0.10, 0.25
# Actual / expected claims outside this band means the model's level is off.
AE_BAND = (0.90, 1.10)
# Gini below 80% of its validation value means the model ranks risks noticeably worse.
# (Deviance is reported too, but it rises on any riskier book, so it is not used as an alert.)
GINI_ALERT_RATIO = 0.80

_EPS = 1e-6


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population stability index of `current` against `reference`.

    Numeric columns are binned on the reference deciles; categorical columns compare category shares,
    so a category that never appeared in the reference still counts as drift.
    """
    if pd.api.types.is_numeric_dtype(reference):
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_share = np.histogram(reference, edges)[0] / len(reference)
        cur_share = np.histogram(current, edges)[0] / len(current)
    else:
        ref_counts = reference.astype(str).value_counts(normalize=True)
        cur_counts = current.astype(str).value_counts(normalize=True)
        cats = ref_counts.index.union(cur_counts.index)
        ref_share = ref_counts.reindex(cats, fill_value=0).to_numpy()
        cur_share = cur_counts.reindex(cats, fill_value=0).to_numpy()
    ref_share = np.clip(ref_share, _EPS, None)
    cur_share = np.clip(cur_share, _EPS, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_status(value: float) -> str:
    if value >= PSI_ALERT:
        return "alert"
    return "watch" if value >= PSI_WATCH else "ok"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for col in models.FEATURES:
        value = psi(reference[col], current[col])
        out[col] = {"psi": round(value, 4), "status": psi_status(value)}
    return out


def performance_report(model, current: pd.DataFrame, baseline_gini: float) -> dict:
    """Needs claims to have come in for the batch; in production this runs once the period's claims are known."""
    y = models.frequency(current).to_numpy()
    w = current["exposure"].to_numpy()
    pred = model.predict(current[models.FEATURES])
    expected = float((pred * w).sum())
    actual = float((y * w).sum())
    ae = actual / expected
    deviance = mean_poisson_deviance(y, np.clip(pred, 1e-9, None), sample_weight=w)
    gini = models.gini(y, pred, w)
    reasons = []
    if not AE_BAND[0] <= ae <= AE_BAND[1]:
        reasons.append(f"actual/expected claims {ae:.2f} outside {AE_BAND[0]:.2f}-{AE_BAND[1]:.2f}")
    if baseline_gini > 0 and gini < GINI_ALERT_RATIO * baseline_gini:
        reasons.append(f"Gini {gini:.3f} below {GINI_ALERT_RATIO:.0%} of validation {baseline_gini:.3f}")
    return {
        "actual_claims": round(actual, 1),
        "expected_claims": round(expected, 1),
        "actual_to_expected": round(ae, 3),
        "gini": round(gini, 4),
        "poisson_deviance": round(deviance, 4),
        "status": "alert" if reasons else "ok",
        "reasons": reasons,
    }


def summarize(drift: dict[str, dict], performance: dict) -> dict:
    drifted = [col for col, v in drift.items() if v["status"] == "alert"]
    watched = [col for col, v in drift.items() if v["status"] == "watch"]
    if performance["status"] == "alert":
        status, action = "alert", "Model output is off; investigate and consider retraining."
    elif drifted:
        status, action = "watch", "Inputs shifted but predictions still match claims; keep monitoring."
    elif watched:
        status, action = "watch", "Small input shifts; no action needed yet."
    else:
        status, action = "ok", "No action needed."
    return {"status": status, "action": action, "drifted_features": drifted, "watched_features": watched}
