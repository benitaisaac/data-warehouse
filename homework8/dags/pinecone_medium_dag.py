# pinecone_medium_dag.py
from datetime import datetime
import os
import json
import time
import math
import pandas as pd

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# -------------- Helpers & Config (import-safe) --------------
def _v(key, default=None, aliases=None):
    """Get an Airflow Variable with env fallback and optional alias keys."""
    aliases = aliases or []
    # Try Variables by key or aliases
    for k in [key] + aliases:
        try:
            return Variable.get(k)
        except Exception:
            pass
    # Env fallback (AIRFLOW_VAR_<KEY> style)
    env_key = f"AIRFLOW_VAR_{key}".upper()
    if os.getenv(env_key):
        return os.getenv(env_key)
    return default

# Vars (supports lowercase professor keys and your uppercase ones)
INDEX_NAME = _v("pinecone_index_name", default="semantic-search-fast",
                aliases=["PINECONE_INDEX_NAME"])
API_KEY    = _v("pinecone_api_key", aliases=["PINECONE_API_KEY"])
CLOUD      = _v("pinecone_cloud", default="aws", aliases=["PINECONE_CLOUD"])
REGION     = _v("pinecone_region", default="us-east-1", aliases=["PINECONE_REGION"])

DATA_URL   = "https://s3-geospatial.s3.us-west-2.amazonaws.com/medium_data.csv"
DATA_DIR   = "/tmp/medium_data"
RAW_CSV    = f"{DATA_DIR}/medium_data.csv"
PREP_PARQ  = f"{DATA_DIR}/medium_preprocessed.parquet"   # for speed/robustness
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM  = 384
BATCH_SIZE = 100  # show batch progress in logs

def _require_api_key():
    if not API_KEY:
        raise RuntimeError(
            "Missing Pinecone API key. Set Airflow Variable 'pinecone_api_key' "
            "(or 'PINECONE_API_KEY') or env AIRFLOW_VAR_PINECONE_API_KEY."
        )

# -------------- DAG --------------
with DAG(
    dag_id="pinecone_medium_demo",
    schedule=None,  # run on demand (you trigger it)
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["pinecone", "hw8"],
) as dag:

    # -------- download_data --------
    def _download_data():
        import requests
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"[download_data] Downloading from {DATA_URL}")
        r = requests.get(DATA_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(RAW_CSV, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        # Count lines for screenshot
        with open(RAW_CSV, "r") as f:
            line_count = sum(1 for _ in f)
        print(f"[download_data] ✅ Saved to {RAW_CSV}")
        print(f"[download_data] ✅ Downloaded file has {line_count} lines")
        return RAW_CSV

    # -------- preprocess_data --------
    def _preprocess():
        print(f"[preprocess_data] Reading {RAW_CSV}")
        df = pd.read_csv(RAW_CSV)
        start_rows = len(df)
        # Clean + metadata like professor
        df["title"] = df["title"].astype(str).fillna("")
        df["subtitle"] = df["subtitle"].astype(str).fillna("")
        df["metadata"] = (df["title"] + " " + df["subtitle"]).apply(
            lambda t: {"title": t}
        )
        if "id" not in df.columns:
            df = df.reset_index(drop=True)
            df["id"] = df.index.astype(str)
        df.to_parquet(PREP_PARQ, index=False)
        print(f"[preprocess_data] ✅ Rows in: {start_rows}, rows out: {len(df)}")
        print(f"[preprocess_data] ✅ Saved preprocessed file to {PREP_PARQ}")
        # Show a tiny sample for the screenshot
        print("[preprocess_data] Sample rows:")
        print(df[["id", "metadata"]].head(3).to_string(index=False))

    # -------- create_index --------
    def _create_index():
        from pinecone import Pinecone, ServerlessSpec
        _require_api_key()
        pc = Pinecone(api_key=API_KEY)
        print(f"[create_index] Using cloud={CLOUD}, region={REGION}, index={INDEX_NAME}")
        names = [i["name"] for i in pc.list_indexes()]
        if INDEX_NAME in names:
            print(f"[create_index] Deleting existing index: {INDEX_NAME}")
            pc.delete_index(INDEX_NAME)
        spec = ServerlessSpec(cloud=CLOUD, region=REGION)
        print(f"[create_index] Creating index (dim={EMBED_DIM}, metric=dotproduct)")
        pc.create_index(name=INDEX_NAME, dimension=EMBED_DIM, metric="dotproduct", spec=spec)
        # Wait until ready (explicit log ticks)
        while True:
            d = pc.describe_index(INDEX_NAME)
            if d.status.get("ready"):
                break
            print("[create_index] …waiting for index to be ready")
            time.sleep(1)
        print(f"[create_index] ✅ Index '{INDEX_NAME}' is ready")

    # -------- embed_and_upsert --------
    def _embed_and_upsert():
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone
        _require_api_key()
        print(f"[embed_and_upsert] Loading preprocessed data: {PREP_PARQ}")
        df = pd.read_parquet(PREP_PARQ)
        total = len(df)
        print(f"[embed_and_upsert] Rows to upsert: {total}")
        model = SentenceTransformer(MODEL_NAME)
        pc = Pinecone(api_key=API_KEY)
        index = pc.Index(INDEX_NAME)

        # Batch for visible progress
        batches = math.ceil(total / BATCH_SIZE)
        for b in range(batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, total)
            batch = df.iloc[start:end].copy()
            titles = [m["title"] for m in batch["metadata"]]
            print(f"[embed_and_upsert] Encoding batch {b+1}/{batches} (rows {start}..{end-1})")
            vecs = model.encode(titles)
            # Prepare upsert payload (id, values, metadata)
            upsert_data = []
            for j, (_, row) in enumerate(batch.iterrows()):
                upsert_data.append({
                    "id": str(row["id"]),
                    "values": vecs[j].tolist(),
                    "metadata": row["metadata"],
                })
            index.upsert(upsert_data)
            print(f"[embed_and_upsert] ✅ Upserted batch {b+1}/{batches} ({end-start} vectors)")
        print("[embed_and_upsert] ✅ All embeddings upserted")

    # -------- query_example --------
    def _query_example():
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone
        _require_api_key()
        pc = Pinecone(api_key=API_KEY)
        index = pc.Index(INDEX_NAME)
        model = SentenceTransformer(MODEL_NAME)
        query = "what is ethics in AI"
        print(f"[query_example] Query: {query!r}")
        qvec = model.encode(query).tolist()
        res = index.query(vector=qvec, top_k=5, include_metadata=True, include_values=False)
        matches = res.get("matches", [])
        print("[query_example] ✅ Top 5 results:")
        for i, m in enumerate(matches, 1):
            title = (m.get("metadata") or {}).get("title", "")[:90]
            score = m.get("score", 0.0)
            print(f"  {i}. score={score:.4f}  title={title}")

    # Create operators with your original task IDs
    t1 = PythonOperator(task_id="download_data", python_callable=_download_data)
    t2 = PythonOperator(task_id="preprocess_data", python_callable=_preprocess)
    t3 = PythonOperator(task_id="create_index", python_callable=_create_index)
    t4 = PythonOperator(task_id="embed_and_upsert", python_callable=_embed_and_upsert)
    t5 = PythonOperator(task_id="query_example", python_callable=_query_example)

    t1 >> t2 >> t3 >> t4 >> t5
