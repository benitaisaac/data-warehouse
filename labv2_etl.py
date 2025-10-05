# dags/two_stock_simple.py

from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook


# ---------------------------
# Snowflake connection (via Airflow Connection)
# ---------------------------
def return_snowflake_conn():
    hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
    conn = hook.get_conn()
    return conn.cursor()  


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

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date.astype(str)

        out[sym] = df.to_dict(orient="records")

    return out



@task
def transform(raw):
    """
    Turn raw yfinance records into rows ready for INSERT:
    [SYMBOL, DATE, OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME]
    """
    rows = []
    for sym, recs in raw.items():
        if not recs:
            continue

        df = pd.DataFrame(recs)

        needed = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if not needed.issubset(df.columns):
            continue
        df = df.dropna(subset=["Close"])
        df = df.rename(columns={"Date": "DATE","Open": "OPEN","High": "HIGH","Low": "LOW",
            "Close": "CLOSE","Adj Close": "ADJ_CLOSE","Volume": "VOLUME", })

        df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None).dt.date.astype(str)

        df = df[(df["LOW"] <= df["HIGH"]) & (df["VOLUME"] >= 0)]

        df["SYMBOL"] = sym
        df = df[["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJ_CLOSE", "VOLUME"]]

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
        cur.execute(ddl)          
        cur.execute("BEGIN")      
        cur.execute(f"TRUNCATE TABLE {target_table}")  

        if rows:
            cur.executemany(insert_sql, rows)       

        cur.execute("COMMIT")    
        return len(rows)
    except Exception:
        cur.execute("ROLLBACK")   
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass


# ---------------------------
# DAG 
# ---------------------------
with DAG(
    dag_id='TwoStockV2',
    start_date=datetime(2024, 9, 21),
    catchup=False,
    tags=['ETL'],
    schedule='30 2 * * *',           
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:

    target_table = "RAW.TWO_STOCK_V2"

    raw_data = extract()
    tidy_rows = transform(raw_data)
    inserted = load(tidy_rows, target_table=target_table)

    raw_data >> tidy_rows >> inserted
