#!/usr/bin/env python3
"""Score three simulated batches against the trained model and write output/monitoring_report.json.

The batches stand in for future quarters of policies:
  steady      - more of the same portfolio (should pass)
  mix_shift   - a younger, higher-bonus-malus book (inputs drift, but the model should still be right)
  claims_trend - same policies, 20% more claims (inputs look normal, but the model under-predicts)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from claims_model import data, models, monitoring

OUT = data.ROOT / "output"
MODEL_PATH = OUT / "model.joblib"


def load_split():
    conn = sqlite3.connect(OUT / "policies.sqlite")
    df = data.model_table(conn)
    conn.close()
    return train_test_split(df, test_size=0.25, random_state=42)


def batches(test, model, rng):
    steady = test.sample(frac=0.5, random_state=1)

    weight = np.where(test["driv_age"] < 30, 5.0, 1.0) * np.where(test["bonus_malus"] >= 100, 3.0, 1.0)
    mix_shift = test.sample(n=len(test) // 2, replace=True, weights=weight, random_state=2)

    trend = test.sample(frac=0.5, random_state=3).copy()
    expected = model.predict(trend[models.FEATURES]) * trend["exposure"]
    trend["claims"] = trend["claims"] + rng.poisson(0.2 * expected)

    return {"steady": steady, "mix_shift": mix_shift, "claims_trend": trend}


def main() -> None:
    train, test = load_split()
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = models.fit(models.gradient_boosting(), train)
        joblib.dump(model, MODEL_PATH)
    validation = models.evaluate(model, test)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "Gradient boosting (Poisson loss)",
        "validation": {k: round(v, 4) for k, v in validation.items()},
        "thresholds": {
            "psi_watch": monitoring.PSI_WATCH,
            "psi_alert": monitoring.PSI_ALERT,
            "actual_to_expected_band": list(monitoring.AE_BAND),
            "gini_alert_ratio": monitoring.GINI_ALERT_RATIO,
        },
        "batches": {},
    }
    rng = np.random.default_rng(7)
    for name, batch in batches(test, model, rng).items():
        drift = monitoring.drift_report(train, batch)
        perf = monitoring.performance_report(model, batch, validation["gini"])
        report["batches"][name] = {
            "policies": len(batch),
            **monitoring.summarize(drift, perf),
            "performance": perf,
            "drift": drift,
        }
    # Flat list for tools that loop over rows (e.g. a Power Automate "Apply to each" or HTML table).
    report["summary"] = [
        {
            "batch": name,
            "status": b["status"],
            "action": b["action"],
            "actual_to_expected": b["performance"]["actual_to_expected"],
            "drifted_features": ", ".join(b["drifted_features"]) or "none",
        }
        for name, b in report["batches"].items()
    ]
    statuses = [b["status"] for b in report["batches"].values()]
    report["overall_status"] = "alert" if "alert" in statuses else "watch" if "watch" in statuses else "ok"

    (OUT / "monitoring_report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, b in report["batches"].items():
        perf = b["performance"]
        print(f"{name:13s} {b['status']:6s} A/E={perf['actual_to_expected']:.3f} "
              f"gini={perf['gini']:.3f} drifted={b['drifted_features']}")


if __name__ == "__main__":
    main()
