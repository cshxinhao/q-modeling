import sqlite3
import json
from typing import Dict, Optional, Any
import cupy as cp
from joblib import load, dump
from pathlib import Path
import pandas as pd

# Model
from xgboost import XGBRegressor


class ModelMetaManager:
    def __init__(self, db_path: str = "msr.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_meta (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                hyper_params TEXT NOT NULL,
                train_params TEXT NOT NULL,
                feature_config TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
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
