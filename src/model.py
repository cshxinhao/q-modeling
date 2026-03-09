import sqlite3
import json
from typing import Dict, Optional, Any
import cupy as cp
from joblib import load, dump
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime


# Model
from xgboost import XGBRegressor


# ---------------------------------------------------
# Model Meta
# ---------------------------------------------------


def model_id(
    train_start: str,
    train_end: str,
    label: str,
    model_params: dict,
):
    base = f"{label}_{train_start}_{train_end}"
    h = hashlib.sha1(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:8]
    return f"{base}_{h}"


def model_files(
    train_start: str,
    train_end: str,
    label: str,
    model_params: dict,
    save_dir: str,
):
    mid = model_id(train_start, train_end, label, model_params)
    model_dir = Path(save_dir)
    joblib_path = model_dir / f"model_{mid}.joblib"
    meta_path = model_dir / f"model_{mid}.meta.json"
    pred_path = model_dir / f"pred_{mid}.parquet"
    pool_path = model_dir / f"stock_pool_{mid}.parquet"
    return joblib_path, meta_path, pred_path, pool_path


def _save_meta(
    meta_path: Path,
    cfg: dict,
    n_samples: int,
    n_features: int,
    selected_features: list[str],
):
    try:
        import importlib.metadata as md

        versions = {}
        for p in ["xgboost", "pandas", "numpy", "streamlit"]:
            try:
                versions[p] = md.version(p)
            except Exception:
                versions[p] = None
    except Exception:
        versions = {}
    payload = {
        "cfg": cfg,
        "trained_at": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_features": n_features,
        "selected_features": list(selected_features),
        "versions": versions,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class ModelMetaManager:
    def __init__(self, db_path: str = "msr.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_meta (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                hyper_params TEXT NOT NULL,
                train_params TEXT NOT NULL,
                feature_config TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def register_model(
        self,
        model_id: str,
        model_type: str,
        hyper_params: Dict[str, Any],
        train_params: Dict[str, Any],
        feature_config: Any,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO model_meta (model_id, model_type, hyper_params, train_params, feature_config)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    model_type,
                    json.dumps(hyper_params),
                    json.dumps(train_params),
                    json.dumps(feature_config),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Model with ID '{model_id}' already exists.")
        finally:
            conn.close()

    def get_model_meta(self, model_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM model_meta WHERE model_id = ?", (model_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "model_id": row["model_id"],
                "model_type": row["model_type"],
                "hyper_params": json.loads(row["hyper_params"]),
                "train_params": json.loads(row["train_params"]),
                "feature_config": json.loads(row["feature_config"]),
                "created_at": row["created_at"],
            }
        return None

    def list_models(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM model_meta", conn)
        conn.close()
        return df


# ---------------------------------------------------
# Training Utils
# ---------------------------------------------------


def train_model(X: pd.DataFrame, y: pd.Series, params: dict):
    reg = XGBRegressor(**params)
    reg.fit(cp.array(X), cp.array(y))
    reg.selected_features = list(X.columns)
    return reg


def save_model(model, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    dump(model, filename)


def load_model(filename: Path):
    return load(filename)


def predict(model, features: pd.DataFrame):
    X = features[model.selected_features]
    pred = pd.Series(model.predict(cp.array(X)), index=X.index, name="pred")
    return pred
