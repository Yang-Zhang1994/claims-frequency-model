#!/usr/bin/env python3
"""Load the raw policies into Snowflake, rebuild the modelling table there, and check it matches SQLite.

Credentials come from environment variables (never commit them):
  SNOWFLAKE_ACCOUNT    account identifier, e.g. abc12345.ca-central-1.aws
  SNOWFLAKE_USER
  SNOWFLAKE_PRIVATE_KEY_PATH   path to a PKCS#8 .p8 key registered on the user (preferred), or
  SNOWFLAKE_PASSWORD           with SNOWFLAKE_AUTHENTICATOR=username_password_mfa if MFA is on
  SNOWFLAKE_ROLE       optional, defaults to the user's default role

Run from the repo root:
  pip install -r snowflake/requirements.txt
  python snowflake/load_to_snowflake.py
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claims_model import data  # noqa: E402

HERE = Path(__file__).resolve().parent


def statements(sql: str) -> list[str]:
    """Split a .sql file into statements: drop "--" comment lines, then split on semicolons."""
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [stmt.strip() for stmt in code.split(";") if stmt.strip()]


def connect():
    import snowflake.connector

    params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "role": os.environ.get("SNOWFLAKE_ROLE"),
    }
    key_path = os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH")
    if key_path:
        params["private_key_file"] = key_path
    else:
        params["password"] = os.environ["SNOWFLAKE_PASSWORD"]
        params["authenticator"] = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "snowflake")
    return snowflake.connector.connect(**{k: v for k, v in params.items() if v})


def main() -> None:
    from snowflake.connector.pandas_tools import write_pandas

    raw = data.fetch_raw()
    for col in raw.select_dtypes(include="category"):
        raw[col] = raw[col].astype(str)
    raw.columns = [c.upper() for c in raw.columns]

    conn = connect()
    cur = conn.cursor()
    try:
        for stmt in statements((HERE / "setup.sql").read_text()):
            cur.execute(stmt)
        ok, _, rows, _ = write_pandas(
            conn, raw, "POLICIES", database="CLAIMS", schema="RAW",
            auto_create_table=True, overwrite=True, quote_identifiers=False,
        )
        print(f"loaded {rows:,} rows into CLAIMS.RAW.POLICIES (ok={ok})")
        for stmt in statements((HERE / "transform.sql").read_text()):
            cur.execute(stmt)

        snowflake_quality = dict(cur.execute("SELECT CHECK_NAME, VALUE FROM CLAIMS.ANALYTICS.DATA_QUALITY").fetchall())
        model_rows = cur.execute("SELECT COUNT(*), SUM(CLAIMS) FROM CLAIMS.ANALYTICS.MODEL_TABLE").fetchone()
    finally:
        cur.close()
        conn.close()

    # Reconcile against the local SQLite pipeline so both platforms agree before anyone uses the table.
    local = sqlite3.connect(":memory:")
    data.stage(data.fetch_raw(), local)
    local_quality = data.quality_report(local)
    local_rows = local.execute("SELECT COUNT(*), SUM(claims) FROM model_table").fetchone()
    local.close()

    mismatches = [k for k, v in local_quality.items() if int(snowflake_quality.get(k, -1)) != v]
    if tuple(model_rows) != tuple(local_rows):
        mismatches.append(f"model_table rows/claims {tuple(model_rows)} vs {tuple(local_rows)}")
    print("data quality:", snowflake_quality)
    print("model_table rows, claims:", tuple(model_rows))
    if mismatches:
        sys.exit("MISMATCH vs SQLite: " + "; ".join(mismatches))
    print("Snowflake matches the SQLite pipeline.")


if __name__ == "__main__":
    main()
