from airflow import DAG
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime
import logging

# Use the connection you already fixed
CONN_ID = "snowflake_conn"
RAW_SCHEMA = "raw"
ANALYTICS_SCHEMA = "analytics"

def return_snowflake_cursor():
    hook = SnowflakeHook(snowflake_conn_id=CONN_ID)
    conn = hook.get_conn()
    return conn.cursor()

@task
def run_ctas(schema, table, select_sql, primary_key=None):
    """
    Create temp table via CTAS, check PK uniqueness, then SWAP into target.
    """
    cur = return_snowflake_cursor()
    try:
        logging.info(f"Building {schema}.{table}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

        # Build temp table
        sql = f"CREATE OR REPLACE TABLE {schema}.temp_{table} AS {select_sql}"
        logging.info(sql)
        cur.execute(sql)

        # Optional PK uniqueness check
        if primary_key:
            check_sql = f"""
              SELECT {primary_key}, COUNT(1) AS cnt
              FROM {schema}.temp_{table}
              GROUP BY 1
              ORDER BY 2 DESC
              LIMIT 1
            """
            logging.info(check_sql)
            cur.execute(check_sql)
            row = cur.fetchone()
            if row and int(row[1]) > 1:
                raise Exception(f"Primary key uniqueness failed: {row}")

        # Ensure target table exists (empty structure)
        create_if_missing = f"""
          CREATE TABLE IF NOT EXISTS {schema}.{table} AS
          SELECT * FROM {schema}.temp_{table} WHERE 1=0
        """
        cur.execute(create_if_missing)

        # 4) Fast swap
        cur.execute(f"ALTER TABLE {schema}.{table} SWAP WITH {schema}.temp_{table}")

    finally:
        cur.close()

with DAG(
    dag_id="BuildELT_CTAS",
    start_date=datetime(2025, 10, 2),
    catchup=False,
    tags=["ELT"],
    schedule=None,   # trigger manually
) as dag:

    # Join RAW → ANALYTICS.session_summary
    select_sql = f"""
    WITH base AS (
      SELECT
        u.userId,
        u.sessionId,
        u.channel,
        s.ts
      FROM {RAW_SCHEMA}.user_session_channel u
      JOIN {RAW_SCHEMA}.session_timestamp s
        ON u.sessionId = s.sessionId
    )
    --Results ensure primary key uniqueness
    SELECT userId, sessionId, channel, ts
    FROM (
      SELECT
        b.*,
        ROW_NUMBER() OVER (PARTITION BY b.sessionId ORDER BY b.ts DESC) AS rn
      FROM base b
    )
    WHERE rn = 1
    """

    run_ctas(
        schema=ANALYTICS_SCHEMA,
        table="session_summary",
        select_sql=select_sql,
        primary_key="sessionId"
    )
