from datetime import datetime, timedelta
import requests
import pandas as pd

from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook

DAG_ID = "stock_to_snowflake"
SNOWFLAKE_CONN_ID = "snowflake_conn"   
TARGET_TABLE = 'USER_DB_COBRA.RAW.AMAZON_STOCK'      
SYMBOL = "AMZN"                        

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def return_snowflake_conn():
    hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
    conn = hook.get_conn()
    return conn.cursor()


with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2024, 1, 1),
    schedule= '30 2 * * *',            # Set up schedule       
    catchup=False,
    default_args=default_args,
    tags=["hw4", "alpha_vantage", "snowflake"],
) as dag:

    @task
    def extract_prices(symbol: str) -> list[dict]:
        api_key = Variable.get("ALPHAVANTAGE_API_KEY")

        url = (
            "https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=compact"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()

        # Debug 
        # print("Alpha Vantage response keys:", list(payload.keys()))

        series = payload.get("Time Series (Daily)")
        if not series:
            # surfaces rate-limit / premium / bad-key messages
            raise ValueError(
                f"No 'Time Series (Daily)' found. Message: "
                f"{payload.get('Information') or payload.get('Note') or payload.get('Error Message')}"
            )

        rows = []
        for d, vals in series.items():
            rows.append({
                "symbol": symbol,
                "date": d,
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": int(vals["5. volume"]),
            })
        return rows


    @task
    def transform(rows: list[dict]) -> list[dict]:
        # clean-up
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # Return a list of dicts 
        return df.to_dict(orient="records")

    @task
    def load_full_refresh(records: list[dict]) -> None:
        """
        Implements full refresh using an explicit SQL transaction (+3).
        """
        if not records:
            return

        cur = return_snowflake_conn() 
        try:
            # Create table (idempotent)
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            SYMBOL VARCHAR(16) NOT NULL,
            "DATE" DATE NOT NULL,
            OPEN   NUMBER(18,6),
            HIGH   NUMBER(18,6),
            LOW    NUMBER(18,6),
            CLOSE  NUMBER(18,6),
            VOLUME NUMBER(38,0),
            PRIMARY KEY (SYMBOL, "DATE")
            );""")

            # Begin transaction
            cur.execute("BEGIN;")

            # Full refresh
            cur.execute(f"TRUNCATE TABLE {TARGET_TABLE};")

            # Bulk insert (same SQL you had)
            insert_sql = f"""
                INSERT INTO {TARGET_TABLE}
                (SYMBOL, "DATE", OPEN, HIGH, LOW, CLOSE, VOLUME)
                VALUES (%(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s)
            """
            cur.executemany(insert_sql, records)

            # Commit transaction
            cur.execute("COMMIT;")
        except Exception:
            cur.execute("ROLLBACK;")
            raise
        finally:
            # Close cursor and its underlying connection
            try:
                conn = getattr(cur, "connection", None)
                cur.close()
                if conn:
                    conn.close()
            except Exception:
                pass


    
    load_full_refresh(transform(extract_prices(SYMBOL)))

