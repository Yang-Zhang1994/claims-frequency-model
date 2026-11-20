"""Exposure-weighted claim-frequency models and the metrics used to compare them."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder

CATEGORICAL = ["area", "veh_brand", "veh_gas", "region"]
BINNED = ["veh_age", "driv_age"]
NUMERIC = ["veh_power", "bonus_malus", "density"]
FEATURES = CATEGORICAL + BINNED + NUMERIC


def frequency(df: pd.DataFrame) -> pd.Series:
    """Claims per policy-year: the target. Exposure is passed separately as the sample weight."""
    return df["claims"] / df["exposure"]


def baseline() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def poisson_glm() -> object:
    # A GLM is linear in log-frequency, so ages are binned to let the curve bend,
    # and density is logged because its effect is roughly proportional.
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bins", KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile"), BINNED),
        ("num", "passthrough", ["veh_power", "bonus_malus"]),
        ("log_density", FunctionTransformer(np.log), ["density"]),
    ])
    return make_pipeline(prep, PoissonRegressor(alpha=1e-6, solver="newton-cholesky", max_iter=300))


def gradient_boosting() -> object:
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL)],
        remainder="passthrough",
    )
    model = HistGradientBoostingRegressor(
        loss="poisson", max_leaf_nodes=64, learning_rate=0.05, max_iter=400,
        categorical_features=list(range(len(CATEGORICAL))), random_state=0,
    )
    return make_pipeline(prep, model)


def fit(model, df: pd.DataFrame):
    last_step = model.steps[-1][0] + "__sample_weight" if hasattr(model, "steps") else "sample_weight"
    return model.fit(df[FEATURES], frequency(df), **{last_step: df["exposure"]})


def gini(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """Exposure-weighted Gini of the Lorenz curve, ranking policies from safest to riskiest prediction.

    0 means the model ranks no better than random; higher is better at separating risks.
    """
    order = np.argsort(y_pred)
    claims = (y_true * exposure)[order]
    cum_exposure = np.cumsum(exposure[order]) / exposure.sum()
    cum_claims = np.cumsum(claims) / claims.sum()
    cum_exposure = np.concatenate([[0.0], cum_exposure])
    cum_claims = np.concatenate([[0.0], cum_claims])
    area_under_lorenz = np.trapezoid(cum_claims, cum_exposure)
    return 1 - 2 * area_under_lorenz


def decile_calibration(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> pd.DataFrame:
    """Observed vs predicted frequency in 10 equal-exposure bands of predicted risk."""
    order = np.argsort(y_pred)
    cum = np.cumsum(exposure[order]) / exposure.sum()
    band = np.minimum((cum * 10).astype(int), 9) + 1
    frame = pd.DataFrame({
        "decile": band,
        "exposure": exposure[order],
        "observed_claims": (y_true * exposure)[order],
        "predicted_claims": (y_pred * exposure)[order],
    })
    out = frame.groupby("decile").sum()
    out["observed_freq"] = out["observed_claims"] / out["exposure"]
    out["predicted_freq"] = out["predicted_claims"] / out["exposure"]
    return out[["exposure", "observed_freq", "predicted_freq"]].round(4)


def evaluate(model, df: pd.DataFrame) -> dict[str, float]:
    y = frequency(df).to_numpy()
    w = df["exposure"].to_numpy()
    pred = model.predict(df[FEATURES])
    return {
        "poisson_deviance": mean_poisson_deviance(y, np.clip(pred, 1e-9, None), sample_weight=w),
        "gini": gini(y, pred, w),
        "predicted_to_observed_claims": float((pred * w).sum() / (y * w).sum()),
    }
