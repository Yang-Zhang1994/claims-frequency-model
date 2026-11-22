"""MCP server: lets an AI agent (Claude Code, Claude Desktop, any MCP client) score and explain policies.

Run:  python -m claims_model.mcp_server            (stdio; models from output/models.joblib)
      CLAIMS_MODELS=/path/to/models.joblib python -m claims_model.mcp_server

The agent never sees training data. It can only call these tools, whose inputs are validated against
the ranges the models were trained on.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from claims_model import data, models

Area = Literal["A", "B", "C", "D", "E", "F"]
Brand = Literal["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
Region = Literal["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53",
                 "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"]

MODEL_NAMES = {"glm": "Poisson GLM", "gradient_boosting": "Gradient boosting",
               "neural_net": "Neural net (PyTorch)", "cann": "GLM + neural correction (CANN)"}


class Policy(BaseModel):
    """A French motor third-party liability policy, in the units of the freMTPL2 data."""

    area: Area = Field(description="Density class of the driver's town, A (rural) to F (urban centre)")
    veh_brand: Brand
    veh_gas: Literal["Diesel", "Regular"]
    region: Region = Field(description="French region code (pre-2016 regions)")
    veh_power: int = Field(ge=4, le=15, description="Fiscal horsepower (values above 12 are treated as 12)")
    veh_age: int = Field(ge=0, le=100, description="Vehicle age in years (above 20 treated as 20)")
    driv_age: int = Field(ge=18, le=100, description="Driver age in years (above 90 treated as 90)")
    bonus_malus: int = Field(ge=50, le=230, description="French bonus-malus coefficient: 50 is the best record, 100 is a new driver, above 100 means past claims (above 150 treated as 150)")
    density: float = Field(gt=0, le=30000, description="Inhabitants per km² where the driver lives")

    def frame(self) -> pd.DataFrame:
        row = self.model_dump()
        # The same caps the SQL prep applies to the training data.
        row.update(veh_power=min(self.veh_power, 12), veh_age=min(self.veh_age, 20),
                   driv_age=min(self.driv_age, 90), bonus_malus=min(self.bonus_malus, 150))
        return pd.DataFrame([row])[models.FEATURES]


def build(bundle: dict[str, Any], monitoring_path: Path | None = None) -> MCPServer:
    server = MCPServer("claims-frequency")
    gbm_test = np.asarray(bundle["test_predictions"]["gradient_boosting"])
    exposure = np.asarray(bundle["test_predictions"]["exposure"])
    order = np.argsort(gbm_test)
    cum = np.cumsum(exposure[order]) / exposure.sum()
    # Decile boundaries in predicted frequency, equal exposure per decile, from held-out policies.
    cutoffs = [float(gbm_test[order][np.searchsorted(cum, q)]) for q in np.arange(0.1, 1.0, 0.1)]

    def predict(key: str, X: pd.DataFrame) -> float:
        return float(bundle[key].predict(X)[0])

    @server.tool()
    def score_policy(policy: Policy) -> dict[str, Any]:
        """Predicted claim frequency (claims per policy-year) from every model, and the policy's risk decile
        (1 = safest 10% of the held-out book, 10 = riskiest), based on the gradient boosting model."""
        X = policy.frame()
        preds = {MODEL_NAMES[k]: round(predict(k, X), 4) for k in MODEL_NAMES}
        gbm = preds["Gradient boosting"]
        decile = int(np.searchsorted(cutoffs, gbm, side="right")) + 1
        portfolio = float(bundle["glm"].predict(pd.DataFrame([bundle["reference"]])[models.FEATURES])[0])
        return {"claims_per_year": preds, "risk_decile": decile,
                "vs_typical_policy": round(gbm / predict("gradient_boosting", pd.DataFrame([bundle["reference"]])[models.FEATURES]), 2),
                "typical_policy_glm_frequency": round(portfolio, 4)}

    @server.tool()
    def explain_policy(policy: Policy) -> dict[str, Any]:
        """Why the GLM prices this policy as it does: for each rating factor, the multiplier from changing the
        typical policy's value to this policy's value. The GLM is multiplicative, so the multipliers
        combine exactly: typical frequency × all multipliers = this policy's frequency."""
        X = policy.frame()
        ref = pd.DataFrame([bundle["reference"]])[models.FEATURES]
        base = predict("glm", ref)
        factors = {}
        for col in models.FEATURES:
            changed = ref.copy()
            changed[col] = X[col].iloc[0]
            factors[col] = {"value": X[col].iloc[0].item() if hasattr(X[col].iloc[0], "item") else X[col].iloc[0],
                            "typical": bundle["reference"][col],
                            "multiplier": round(predict("glm", changed) / base, 3)}
        ranked = dict(sorted(factors.items(), key=lambda kv: -abs(np.log(kv[1]["multiplier"]))))
        return {"typical_frequency": round(base, 4), "policy_frequency": round(predict("glm", X), 4), "factors": ranked}

    @server.tool()
    def compare_models() -> list[dict[str, Any]]:
        """Held-out scores for every model: Poisson deviance (lower is better), Gini (higher separates risks
        better) and predicted/observed total claims (near 1.0 means the overall level is right)."""
        return bundle["scores"]

    @server.tool()
    def monitoring_status() -> dict[str, Any]:
        """Latest production-style monitoring report: input drift (PSI) and actual/expected claims per batch."""
        path = monitoring_path or data.ROOT / "output" / "monitoring_report.json"
        if not path.exists():
            return {"status": "unknown", "detail": "No monitoring report yet; run monitor.py."}
        report = json.loads(path.read_text())
        return {"generated_at": report["generated_at"], "overall_status": report["overall_status"],
                "summary": report["summary"]}

    @server.resource("claims://model-card", mime_type="text/markdown")
    def model_card() -> str:
        """What the models are, what they were trained on, and what they must not be used for."""
        return (data.ROOT / "docs" / "model-card.md").read_text()

    return server


def main() -> None:
    path = Path(os.environ.get("CLAIMS_MODELS", data.ROOT / "output" / "models.joblib"))
    build(joblib.load(path)).run()


if __name__ == "__main__":
    main()
