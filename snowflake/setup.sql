-- One-time objects for the claims pipeline (run by load_to_snowflake.py before loading).

-- warehouse that suspends itself after 60 s idle, so a trial account barely spends credits
CREATE WAREHOUSE IF NOT EXISTS CLAIMS_WH
  WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE;

USE WAREHOUSE CLAIMS_WH;

CREATE DATABASE IF NOT EXISTS CLAIMS;

CREATE SCHEMA IF NOT EXISTS CLAIMS.RAW;

CREATE SCHEMA IF NOT EXISTS CLAIMS.ANALYTICS;
