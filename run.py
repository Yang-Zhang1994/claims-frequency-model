#!/usr/bin/env python3
"""Stage data in SQLite, fit three frequency models, and write results to output/."""

from __future__ import annotations

import sqlite3
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from claims_model import data, models

OUT = data.ROOT / "output"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    conn = sqlite3.connect(OUT / "policies.sqlite")
    data.stage(data.fetch_raw(), conn)
    quality = data.quality_report(conn)
    by_bm = pd.read_sql(data.FREQ_BY_BONUS_MALUS_SQL, conn)
    df = data.model_table(conn)
    conn.close()

    train, test = train_test_split(df, test_size=0.25, random_state=42)

    candidates = {
        "Baseline (portfolio mean)": models.baseline(),
        "Poisson GLM": models.poisson_glm(),
        "Gradient boosting (Poisson loss)": models.gradient_boosting(),
    }
    rows, preds = [], {}
    for name, model in candidates.items():
        start = time.perf_counter()
        models.fit(model, train)
        seconds = time.perf_counter() - start
        rows.append({"model": name, "fit_seconds": round(seconds, 1),
                     **{k: round(v, 4) for k, v in models.evaluate(model, test).items()}})
        preds[name] = model.predict(test[models.FEATURES])
    scores = pd.DataFrame(rows)
    # The boosted model is the one monitor.py watches.
    joblib.dump(candidates["Gradient boosting (Poisson loss)"], OUT / "model.joblib")

    y = models.frequency(test).to_numpy()
    w = test["exposure"].to_numpy()
    calib = models.decile_calibration(y, preds["Gradient boosting (Poisson loss)"], w)

    plot_lorenz(y, w, preds)
    plot_calibration(calib)
    write_report(quality, by_bm, len(train), len(test), scores, calib)
    print(scores.to_string(index=False))


def plot_lorenz(y, w, preds) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, pred in preds.items():
        order = np.argsort(pred)
        x = np.concatenate([[0], np.cumsum(w[order]) / w.sum()])
        c = np.concatenate([[0], np.cumsum((y * w)[order]) / (y * w).sum()])
        ax.plot(x, c, label=f"{name} (Gini {models.gini(y, pred, w):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("Cumulative share of exposure (safest first)")
    ax.set_ylabel("Cumulative share of claims")
    ax.set_title("Lorenz curves on held-out policies")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "lorenz.png", dpi=140)
    plt.close(fig)


def plot_calibration(calib: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(calib.index, calib["observed_freq"], "o-", label="Observed")
    ax.plot(calib.index, calib["predicted_freq"], "s--", label="Predicted (gradient boosting)")
    ax.set_xlabel("Predicted-risk decile (equal exposure)")
    ax.set_ylabel("Claims per policy-year")
    ax.set_xticks(calib.index)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=140)
    plt.close(fig)


def write_report(quality, by_bm, n_train, n_test, scores, calib) -> None:
    lines = [
        "# Results",
        "",
        "## Data quality (raw table, SQL)",
        "",
        *[f"- {k}: {v:,}" for k, v in quality.items()],
        "",
        "## Observed frequency by bonus-malus band (SQL)",
        "",
        by_bm.to_markdown(index=False),
        "",
        f"## Held-out scores ({n_train:,} train / {n_test:,} test policies)",
        "",
        "Lower Poisson deviance is better; higher Gini separates risks better; "
        "predicted/observed claims near 1.0 means the total is right.",
        "",
        scores.to_markdown(index=False),
        "",
        "## Calibration by decile (gradient boosting)",
        "",
        calib.to_markdown(),
        "",
    ]
    (OUT / "results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
