# dags/two_stock_simple.py

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from airflow.operators.trigger_dagrun import TriggerDagRunOperator



# ---------------------------
# Snowflake connection (via Airflow Connection)
# ---------------------------
def return_snowflake_conn():
    # Make sure you have a Connection named 'snowflake_conn' in Airflow
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    conn = hook.get_conn()
    return conn.cursor()  # DB-API cursor


# ---------------------------
# EXTRACT
# ---------------------------
@task
def extract():
    """
    Download raw yfinance data and return it as {symbol: [raw-records]}.
    Ensures columns are single-level and JSON-serializable.
    """
    symbols_csv = Variable.get("stock_symbols", default_var="DIS,NFLX")
    lookback_days = int(Variable.get("lookback_days", default_var="180"))
    symbols = [s.strip() for s in symbols_csv.split(",") if s.strip()]

    out = {}
    for sym in symbols:
        df = yf.download(
            sym,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,       # avoid cookie/SQLite issue
            group_by="column",   # ensure non-MultiIndex columns
        )

        if df is None or df.empty:
            out[sym] = []
            continue

        # Bring Date out of index
        df = df.reset_index()

        # If columns are still MultiIndex (some environments), flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        # Make values JSON/XCom-safe: Date -> 'YYYY-MM-DD' strings
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date.astype(str)

        # Convert to plain dicts for XCom
        out[sym] = df.to_dict(orient="records")

    return out



@task
def transform(raw):
    """
    Turn raw yfinance records into rows ready for INSERT:
    [SYMBOL, DATE, OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME]
    """
    import pandas as pd

    rows = []
    for sym, recs in raw.items():
        if not recs:
            continue

        df = pd.DataFrame(recs)

        # make sure expected raw columns exist
        needed = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if not needed.issubset(df.columns):
            # skip symbol if structure unexpected
            continue

        # drop rows with missing Close BEFORE renaming
        df = df.dropna(subset=["Close"])

        # rename to target schema
        df = df.rename(columns={
            "Date": "DATE",
            "Open": "OPEN",
            "High": "HIGH",
            "Low": "LOW",
            "Close": "CLOSE",
            "Adj Close": "ADJ_CLOSE",
            "Volume": "VOLUME",
        })

        # DATE -> 'YYYY-MM-DD' strings for XCom (Snowflake will cast to DATE)
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None).dt.date.astype(str)

        # basic sanity filter
        df = df[(df["LOW"] <= df["HIGH"]) & (df["VOLUME"] >= 0)]

        # add symbol; keep only needed cols in order
        df["SYMBOL"] = sym
        df = df[["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]]

        # replace NaN with None so executemany works cleanly
        df = df.where(pd.notna(df), None)

        rows.extend(df.values.tolist())

    return rows


# ---------------------------
# LOAD
# ---------------------------
@task
def load(rows, target_table: str):
    """
    Creates the table if needed, truncates it (full refresh), then inserts all rows.
    Uses a transaction with COMMIT/ROLLBACK.
    """
    # Table DDL — safe to run every time thanks to IF NOT EXISTS
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {target_table} (
      SYMBOL       VARCHAR(16)   NOT NULL,
      "DATE"       DATE          NOT NULL,
      OPEN         NUMBER(18,6),
      HIGH         NUMBER(18,6),
      LOW          NUMBER(18,6),
      CLOSE        NUMBER(18,6),
      VOLUME       NUMBER(38,0),
      ADJ_CLOSE    NUMBER(18,6),
      SOURCE       VARCHAR(32)    DEFAULT 'yfinance',
      LOAD_TS      TIMESTAMP_NTZ  DEFAULT CURRENT_TIMESTAMP(),
      PRIMARY KEY (SYMBOL, "DATE")
    );
    """

    insert_sql = f"""
    INSERT INTO {target_table}
      (SYMBOL, "DATE", OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    cur = return_snowflake_conn()
    try:
        cur.execute(ddl)           # 1) Make sure table exists
        cur.execute("BEGIN")       # 2) Start transaction
        cur.execute(f"TRUNCATE TABLE {target_table}")  # 3) Full refresh

        if rows:
            cur.executemany(insert_sql, rows)          # 4) Insert data

        cur.execute("COMMIT")      # 5) Commit
        return len(rows)
    except Exception:
        cur.execute("ROLLBACK")    # If anything fails, undo changes
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass


# ---------------------------
# DAG (glue everything together)
# ---------------------------
with DAG(
    dag_id='TwoStockV2',
    start_date=datetime(2024, 9, 21),
    catchup=False,
    tags=['ETL'],
    schedule='30 2 * * *',             # run daily at 02:30
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    # You can change this from the UI (Admin -> Variables)
    target_table = "USER_DB_COBRA.RAW.TWO_STOCK_V2"

    raw_data = extract()
    tidy_rows = transform(raw_data)
    inserted = load(tidy_rows, target_table=target_table)

    trigger_forecast = TriggerDagRunOperator(
        task_id="trigger_forecast",
        trigger_dag_id="TrainPredict",   
        wait_for_completion=False,          
        conf={
            "as_of_date": "{{ ds }}",                       
            "source_table": target_table,                   
            "symbols": "{{ var.value.stock_symbols }}",     # whatever you put in Variables
            "lookback_days": "{{ var.value.lookback_days }}",
        },
    )

    raw_data >> tidy_rows >> inserted >> trigger_forecast
