import sqlite3
import unittest

import numpy as np
import pandas as pd

from claims_model import data, models, monitoring


def toy_raw() -> pd.DataFrame:
    return pd.DataFrame({
        "IDpol": [1.0, 2.0, 3.0, 4.0],
        "ClaimNb": [0, 1, 11, 0],
        "Exposure": [0.5, 1.0, 2.3, 0.0],
        "Area": ["A", "B", "C", "A"],
        "VehPower": [5, 15, 7, 6],
        "VehAge": [2, 30, 4, 1],
        "DrivAge": [40, 99, 25, 30],
        "BonusMalus": [50, 230, 100, 60],
        "VehBrand": ["B1", "B2", "B1", "B3"],
        "VehGas": ["'Regular'", "'Diesel'", "'Regular'", "'Diesel'"],
        "Density": [100, 2000, 50, 10],
        "Region": ["R11", "R24", "R11", "R52"],
    })


class DataPrepTest(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        data.stage(toy_raw(), self.conn)

    def tearDown(self):
        self.conn.close()

    def test_quality_report_flags_bad_rows(self):
        q = data.quality_report(self.conn)
        self.assertEqual(q["rows"], 4)
        self.assertEqual(q["duplicate_policy_ids"], 0)
        self.assertEqual(q["non_positive_exposure"], 1)
        self.assertEqual(q["exposure_over_one_year"], 1)
        self.assertEqual(q["claims_over_four"], 1)

    def test_prep_caps_outliers_and_drops_zero_exposure(self):
        df = data.model_table(self.conn).set_index("policy_id")
        self.assertNotIn(4, df.index)
        self.assertEqual(df.loc[3, "claims"], 4)
        self.assertEqual(df.loc[3, "exposure"], 1.0)
        self.assertEqual(df.loc[2, "bonus_malus"], 150)
        self.assertEqual(df.loc[2, "veh_age"], 20)
        self.assertEqual(df.loc[2, "veh_gas"], "Diesel")


class MetricsTest(unittest.TestCase):
    def test_gini_is_zero_for_constant_prediction_on_uniform_risk(self):
        y = np.ones(100)
        w = np.ones(100)
        self.assertAlmostEqual(models.gini(y, np.ones(100), w), 0.0, places=6)

    def test_gini_rewards_correct_ranking(self):
        y = np.array([0.0, 0.0, 0.0, 4.0])
        w = np.ones(4)
        right = models.gini(y, np.array([0.1, 0.2, 0.3, 0.9]), w)
        wrong = models.gini(y, np.array([0.9, 0.3, 0.2, 0.1]), w)
        self.assertGreater(right, 0.5)
        self.assertLess(wrong, 0)

    def test_decile_calibration_preserves_totals(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(0.1, 1, 1000)
        pred = rng.uniform(0.05, 0.3, 1000)
        y = rng.poisson(pred * w) / w
        calib = models.decile_calibration(y, pred, w)
        self.assertEqual(len(calib), 10)
        observed = (calib["observed_freq"] * calib["exposure"]).sum()
        self.assertAlmostEqual(observed, (y * w).sum(), delta=0.1)


class ModelTest(unittest.TestCase):
    def test_glm_learns_a_known_rate_difference(self):
        rng = np.random.default_rng(1)
        n = 4000
        df = pd.DataFrame({
            "area": rng.choice(["A", "B"], n),
            "veh_brand": "B1", "veh_gas": "Regular", "region": "R11",
            "veh_age": rng.integers(0, 20, n), "driv_age": rng.integers(18, 90, n),
            "veh_power": 6, "bonus_malus": 50, "density": 100,
            "exposure": 1.0,
        })
        rate = np.where(df["area"] == "B", 0.3, 0.1)
        df["claims"] = rng.poisson(rate)
        model = models.fit(models.poisson_glm(), df)
        probe = pd.concat([df.iloc[[0]]] * 2, ignore_index=True)
        probe["area"] = ["A", "B"]
        pred = model.predict(probe[models.FEATURES])
        self.assertAlmostEqual(pred[1] / pred[0], 3.0, delta=0.6)


class MonitoringTest(unittest.TestCase):
    def test_psi_is_near_zero_for_the_same_distribution(self):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(50, 10, 20000))
        cur = pd.Series(rng.normal(50, 10, 20000))
        self.assertLess(monitoring.psi(ref, cur), 0.01)
        self.assertEqual(monitoring.psi_status(monitoring.psi(ref, cur)), "ok")

    def test_psi_flags_a_shifted_distribution(self):
        rng = np.random.default_rng(1)
        ref = pd.Series(rng.normal(50, 10, 20000))
        cur = pd.Series(rng.normal(60, 10, 20000))
        self.assertEqual(monitoring.psi_status(monitoring.psi(ref, cur)), "alert")

    def test_psi_counts_a_category_never_seen_in_training(self):
        ref = pd.Series(["A"] * 500 + ["B"] * 500)
        cur = pd.Series(["A"] * 500 + ["Z"] * 500)
        self.assertEqual(monitoring.psi_status(monitoring.psi(ref, cur)), "alert")

    def test_performance_alerts_when_claims_outrun_predictions(self):
        rng = np.random.default_rng(2)
        n = 20000
        df = pd.DataFrame({
            "area": "A", "veh_brand": "B1", "veh_gas": "Regular", "region": "R11",
            "veh_age": rng.integers(0, 20, n), "driv_age": rng.integers(18, 90, n),
            "veh_power": 6, "bonus_malus": 50, "density": 100, "exposure": 1.0,
        })
        df["claims"] = rng.poisson(0.1, n)
        model = models.fit(models.baseline(), df)
        self.assertEqual(monitoring.performance_report(model, df, baseline_gini=0.0)["status"], "ok")
        more = df.assign(claims=df["claims"] + rng.poisson(0.03, n))
        report = monitoring.performance_report(model, more, baseline_gini=0.0)
        self.assertEqual(report["status"], "alert")
        self.assertGreater(report["actual_to_expected"], 1.1)

    def test_summary_separates_input_drift_from_model_failure(self):
        drift = {"driv_age": {"psi": 0.3, "status": "alert"}, "area": {"psi": 0.01, "status": "ok"}}
        ok_perf = {"status": "ok"}
        bad_perf = {"status": "alert"}
        self.assertEqual(monitoring.summarize(drift, ok_perf)["status"], "watch")
        self.assertEqual(monitoring.summarize(drift, bad_perf)["status"], "alert")
        self.assertEqual(monitoring.summarize({"area": {"psi": 0.0, "status": "ok"}}, ok_perf)["status"], "ok")


class SnowflakeSqlTest(unittest.TestCase):
    def test_sql_files_split_into_the_expected_statements(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("loader", data.ROOT / "snowflake" / "load_to_snowflake.py")
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        setup = loader.statements((data.ROOT / "snowflake" / "setup.sql").read_text())
        transform = loader.statements((data.ROOT / "snowflake" / "transform.sql").read_text())
        self.assertEqual(len(setup), 5)
        self.assertTrue(setup[0].startswith("CREATE WAREHOUSE"))
        self.assertEqual([s.split()[4] for s in transform],
                         ["CLAIMS.ANALYTICS.MODEL_TABLE", "CLAIMS.ANALYTICS.DATA_QUALITY", "CLAIMS.ANALYTICS.FREQ_BY_BONUS_MALUS"])
        self.assertIn("REPLACE(VEHGAS, \'\'\'\', \'\')", transform[0])


if __name__ == "__main__":
    unittest.main()
