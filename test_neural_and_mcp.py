"""Tests for the PyTorch models and the MCP server, on synthetic policies (no download, a few seconds)."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from mcp import Client

from claims_model import models
from claims_model.mcp_server import build
from claims_model.neural import NeuralPoisson


def synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """Policies whose true frequency doubles when bonus_malus >= 100 and is 1.5x in region R11."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area": rng.choice(list("ABCDEF"), n), "veh_brand": rng.choice(["B1", "B2", "B12"], n),
        "veh_gas": rng.choice(["Diesel", "Regular"], n), "region": rng.choice(["R11", "R24", "R82"], n),
        "veh_power": rng.integers(4, 13, n), "veh_age": rng.integers(0, 21, n), "driv_age": rng.integers(18, 91, n),
        "bonus_malus": rng.choice([50, 50, 50, 80, 100, 120], n), "density": rng.integers(1, 20000, n).astype(float),
        "exposure": rng.uniform(0.1, 1.0, n),
    })
    rate = 0.08 * np.where(df["bonus_malus"] >= 100, 2.0, 1.0) * np.where(df["region"] == "R11", 1.5, 1.0)
    df["claims"] = rng.poisson(rate * df["exposure"])
    return df


class NeuralPoissonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 80k policies: with ~20k the validation split holds too few claims and early stopping is noisy.
        cls.train, cls.test = synthetic(80000, 0), synthetic(5000, 1)
        cls.net = models.fit(NeuralPoisson(epochs=15, seed=0), cls.train)

    def test_learns_the_true_risk_ratios(self):
        base = self.test.assign(bonus_malus=50, region="R24")
        p_safe = self.net.predict(base[models.FEATURES]).mean()
        p_malus = self.net.predict(base.assign(bonus_malus=120)[models.FEATURES]).mean()
        p_r11 = self.net.predict(base.assign(region="R11")[models.FEATURES]).mean()
        self.assertAlmostEqual(p_malus / p_safe, 2.0, delta=0.35)
        self.assertAlmostEqual(p_r11 / p_safe, 1.5, delta=0.3)

    def test_beats_the_portfolio_mean_and_keeps_the_total_right(self):
        net = models.evaluate(self.net, self.test)
        mean = models.evaluate(models.fit(models.baseline(), self.train), self.test)
        self.assertLess(net["poisson_deviance"], mean["poisson_deviance"])
        self.assertGreater(net["gini"], 0.1)
        self.assertAlmostEqual(net["predicted_to_observed_claims"], 1.0, delta=0.08)

    def test_balance_correction_matches_training_claims_exactly(self):
        w = self.train["exposure"].to_numpy()
        predicted = (self.net.predict(self.train[models.FEATURES]) * w).sum()
        self.assertAlmostEqual(predicted / self.train["claims"].sum(), 1.0, places=4)

    def test_same_seed_same_model(self):
        again = models.fit(NeuralPoisson(epochs=15, seed=0), self.train)
        np.testing.assert_allclose(again.predict(self.test[models.FEATURES]), self.net.predict(self.test[models.FEATURES]))

    def test_unseen_category_still_scores(self):
        odd = self.test.head(3).assign(veh_brand="B99", region="R00")
        self.assertTrue(np.isfinite(self.net.predict(odd[models.FEATURES])).all())

    def test_cann_starts_from_the_glm(self):
        glm = models.fit(models.poisson_glm(), self.train)
        cann = models.fit(NeuralPoisson(offset=glm, epochs=0), self.train)  # no training: output layer is zero
        X = self.test[models.FEATURES]
        np.testing.assert_allclose(cann.predict(X) / cann.balance_, glm.predict(X), rtol=1e-4)


POLICY = {"area": "E", "veh_brand": "B12", "veh_gas": "Regular", "region": "R11", "veh_power": 7,
          "veh_age": 1, "driv_age": 19, "bonus_malus": 120, "density": 9000}


class McpServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train, test = synthetic(8000, 2), synthetic(2000, 3)
        glm = models.fit(models.poisson_glm(), train)
        gbm = models.fit(models.gradient_boosting(), train)
        cls.bundle = {
            "glm": glm, "gradient_boosting": gbm,
            "neural_net": models.fit(NeuralPoisson(epochs=3), train),
            "cann": models.fit(NeuralPoisson(offset=glm, epochs=3), train),
            "scores": [{"model": "Poisson GLM", **models.evaluate(glm, test)}],
            "reference": {"area": "C", "veh_brand": "B1", "veh_gas": "Regular", "region": "R24", "veh_age": 6.0,
                          "driv_age": 44.0, "veh_power": 6.0, "bonus_malus": 50.0, "density": 400.0},
            "test_predictions": {"gradient_boosting": gbm.predict(test[models.FEATURES]),
                                 "exposure": test["exposure"].to_numpy()},
        }

    def call(self, tool, args=None, monitoring_path=None):
        async def go():
            async with Client(build(self.bundle, monitoring_path)) as client:
                return await client.call_tool(tool, args or {})
        return asyncio.run(go())

    def test_lists_the_four_tools_and_the_model_card(self):
        async def go():
            async with Client(build(self.bundle)) as client:
                tools = [t.name for t in (await client.list_tools()).tools]
                card = await client.read_resource("claims://model-card")
                return tools, card.contents[0].text
        tools, card = asyncio.run(go())
        self.assertEqual(sorted(tools), ["compare_models", "explain_policy", "monitoring_status", "score_policy"])
        self.assertIn("Not for.", card)

    def test_score_policy_returns_every_model_and_a_decile(self):
        out = self.call("score_policy", {"policy": POLICY}).structured_content
        self.assertEqual(set(out["claims_per_year"]), {"Poisson GLM", "Gradient boosting", "Neural net (PyTorch)",
                                                       "GLM + neural correction (CANN)"})
        self.assertIn(out["risk_decile"], range(1, 11))
        self.assertGreater(out["vs_typical_policy"], 1.0)  # malus and R11 both raise risk

    def test_explanation_multipliers_reproduce_the_glm_prediction(self):
        out = self.call("explain_policy", {"policy": POLICY}).structured_content
        product = np.prod([f["multiplier"] for f in out["factors"].values()])
        self.assertAlmostEqual(out["typical_frequency"] * product, out["policy_frequency"], delta=0.002)
        self.assertEqual(next(iter(out["factors"])), max(out["factors"], key=lambda k: abs(np.log(out["factors"][k]["multiplier"]))))

    def test_out_of_range_inputs_are_rejected_with_the_field(self):
        for bad in ({"driv_age": 12}, {"bonus_malus": 40}, {"region": "R99"}, {"density": 0}):
            result = self.call("score_policy", {"policy": {**POLICY, **bad}})
            self.assertTrue(result.is_error, bad)
            self.assertIn(next(iter(bad)), result.content[0].text)

    def test_values_above_the_training_caps_are_capped_like_the_data(self):
        capped = self.call("score_policy", {"policy": {**POLICY, "driv_age": 99}}).structured_content
        at_cap = self.call("score_policy", {"policy": {**POLICY, "driv_age": 90}}).structured_content
        self.assertEqual(capped["claims_per_year"], at_cap["claims_per_year"])

    def test_monitoring_status_reads_the_report_or_says_it_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = self.call("monitoring_status", monitoring_path=Path(tmp) / "none.json").structured_content
            self.assertEqual(missing["status"], "unknown")
            report = Path(tmp) / "r.json"
            report.write_text(json.dumps({"generated_at": "2026-11-22", "overall_status": "watch", "summary": ["x"]}))
            self.assertEqual(self.call("monitoring_status", monitoring_path=report).structured_content["overall_status"], "watch")


if __name__ == "__main__":
    unittest.main()
