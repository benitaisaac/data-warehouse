from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook


def get_snowflake_cursor():
    """Connect to Snowflake via Airflow connection"""
    hook = SnowflakeHook(snowflake_conn_id="snowflake_conn")
    conn = hook.get_conn()
    return conn.cursor()


@task
def create_and_load_tables():
    cursor = get_snowflake_cursor()
    try:
        # SQL Transaction 
        cursor.execute("BEGIN;")

        # Create schema (optional safety)
        cursor.execute("CREATE SCHEMA IF NOT EXISTS raw;")

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw.user_session_channel (
                userId INT NOT NULL,
                sessionId VARCHAR(32) PRIMARY KEY,
                channel VARCHAR(32) DEFAULT 'direct'
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw.session_timestamp (
                sessionId VARCHAR(32) PRIMARY KEY,
                ts TIMESTAMP
            );
        """)

        # Create stage
        cursor.execute("""
            CREATE OR REPLACE STAGE raw.blob_stage
            URL='s3://s3-geospatial/readonly/'
            FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='"');
        """)

        # Copy data from stage to tables
        cursor.execute("""
            COPY INTO raw.user_session_channel
            FROM @raw.blob_stage/user_session_channel.csv;
        """)

        cursor.execute("""
            COPY INTO raw.session_timestamp
            FROM @raw.blob_stage/session_timestamp.csv;
        """)

        cursor.execute("COMMIT;")
        print("Tables created and data loaded successfully.")

    except Exception as e:
        cursor.execute("ROLLBACK;")
        print("Error during ETL:", e)
        raise

    finally:
        cursor.close()


# --- DAG definition ---
with DAG(
    dag_id="etl_raw_sessions_to_snowflake",
    description="Create RAW tables, stage, and copy data from S3",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["etl", "snowflake"],
) as dag:

    create_and_load_tables()
