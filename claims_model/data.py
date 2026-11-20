"""Load the French motor TPL policies, stage them in SQLite, and prepare a modelling table in SQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OPENML_ID = 41214  # freMTPL2freq: 678,013 motor third-party liability policies

RAW_COLUMNS = [
    "IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge",
    "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region",
]

# Caps follow common practice for this dataset: a handful of policies report
# 5-16 claims or more than one year of exposure, which are recording errors
# rather than risk signal.
PREP_SQL = """
SELECT
    CAST(IDpol AS INTEGER)              AS policy_id,
    MIN(ClaimNb, 4)                     AS claims,
    MIN(Exposure, 1.0)                  AS exposure,
    Area                                AS area,
    MIN(VehPower, 12)                   AS veh_power,
    MIN(VehAge, 20)                     AS veh_age,
    MIN(DrivAge, 90)                    AS driv_age,
    MIN(BonusMalus, 150)                AS bonus_malus,
    VehBrand                            AS veh_brand,
    REPLACE(VehGas, '''', '')           AS veh_gas,
    Density                             AS density,
    Region                              AS region
FROM policies
WHERE Exposure > 0
"""

QUALITY_SQL = {
    "rows": "SELECT COUNT(*) FROM policies",
    "duplicate_policy_ids": "SELECT COUNT(*) - COUNT(DISTINCT IDpol) FROM policies",
    "null_cells": "SELECT " + " + ".join(f"SUM({c} IS NULL)" for c in RAW_COLUMNS) + " FROM policies",
    "non_positive_exposure": "SELECT COUNT(*) FROM policies WHERE Exposure <= 0",
    "exposure_over_one_year": "SELECT COUNT(*) FROM policies WHERE Exposure > 1",
    "claims_over_four": "SELECT COUNT(*) FROM policies WHERE ClaimNb > 4",
}

# Observed claim frequency by bonus-malus band, the strongest single rating factor.
FREQ_BY_BONUS_MALUS_SQL = """
SELECT
    CASE
        WHEN bonus_malus < 60  THEN '50-59'
        WHEN bonus_malus < 80  THEN '60-79'
        WHEN bonus_malus < 100 THEN '80-99'
        ELSE '100+'
    END                                       AS bonus_malus_band,
    COUNT(*)                                  AS policies,
    ROUND(SUM(exposure), 0)                   AS exposure_years,
    SUM(claims)                               AS claims,
    ROUND(1.0 * SUM(claims) / SUM(exposure), 4) AS claim_frequency
FROM model_table
GROUP BY bonus_malus_band
ORDER BY MIN(bonus_malus)
"""


def fetch_raw() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=ROOT / "data" / "openml").frame
    return frame[RAW_COLUMNS]


def stage(raw: pd.DataFrame, conn: sqlite3.Connection) -> None:
    out = raw.copy()
    for col in out.select_dtypes(include="category"):
        out[col] = out[col].astype(str)
    out.to_sql("policies", conn, index=False, if_exists="replace")
    conn.execute("DROP TABLE IF EXISTS model_table")
    conn.execute("CREATE TABLE model_table AS " + PREP_SQL)


def quality_report(conn: sqlite3.Connection) -> dict[str, int]:
    return {name: conn.execute(sql).fetchone()[0] for name, sql in QUALITY_SQL.items()}


def model_table(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM model_table", conn)
