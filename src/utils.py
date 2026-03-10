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


# ---------------------------------------------------
# Model Meta
# ---------------------------------------------------


def get_model_id(
    base_model_name: str,
    train_start: str,
    train_end: str,
    label: str,
    model_params: dict,
):
    base = f"{base_model_name}_{label}_{train_start}_{train_end}"
    h = hashlib.sha1(json.dumps(model_params, sort_keys=True).encode()).hexdigest()[:8]
    return f"{base}_{h}"


def get_model_files(model_id: str, save_dir: str):
    model_dir = Path(save_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = model_dir / f"model_{model_id}.joblib"
    meta_path = model_dir / f"model_{model_id}.meta.json"
    pred_path = model_dir / f"pred_{model_id}.parquet"
    pool_path = model_dir / f"stock_pool_{model_id}.parquet"
    return joblib_path, meta_path, pred_path, pool_path


def save_meta(
    train_start: str,
    train_end: str,
    label: str,
    model_params: dict,
    n_samples: int,
    n_features: int,
    selected_features: list[str],
    meta_filename: Path,
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
        "trained_at": datetime.now().isoformat(),
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "label": label,
        "model_params": model_params,
        "n_samples": n_samples,
        "n_features": n_features,
        "selected_features": list(selected_features),
        "versions": versions,
    }
    meta_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_meta(meta_filename: Path) -> dict:
    with open(meta_filename, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


# ---------------------------------------------------
# Forward Returns Calculation
# ---------------------------------------------------


def calc_forward_returns(
    buy_price: pd.DataFrame,
    sell_price: pd.DataFrame,
    horizon: int,
    delay: int,
    adjust: str = None,
):
    """
    * buy_price: pd.DataFrame with index = 'datetime', column = 'symbol'
    * sell_price: pd.DataFrame with index = 'datetime', column = 'symbol'
        The reason to distinguish buy_price and sell_price are:
        1. In markets with limit up/down rules, if the stock is limit up, you can't buy.
        2. The stamp duty is sometimes charged only on the sell side.
    * adjust: str, optional
        The type of return adjustment to apply. Options are:
        None: No adjustment.
        excess_over_eqw: Excess returns over equal-weighted index.
        excess_over_benchmark: Excess returns over benchmark.
        vol_adjusted: Adjust returns over volatility.
        vol_liquid_adjusted: Adjust returns over volatility and liquidity.
        size_adjusted: Adjust returns over size.
    """

    # Assertions
    assert buy_price.index.equals(sell_price.index), (
        "buy_price and sell_price must have the same index"
    )
    assert buy_price.columns.equals(sell_price.columns), (
        "buy_price and sell_price must have the same columns"
    )
    assert adjust in (
        None,  # No adjustment
        "excess_over_eqw",  # Excess returns over equal-weighted index
        "excess_over_benchmark",  # Excess returns over benchmark
        "vol_adjusted",  # Adjust returns over volatility
        "vol_liquid_adjusted",  # Adjust returns over volatility and liquidity
        "size_adjusted",  # Adjust returns over size
    )

    # Calculate raw returns
    forward_returns = sell_price.shift(-horizon - delay) / buy_price.shift(-delay) - 1

    # Adjust returns
    if adjust == "excess_over_eqw":
        forward_returns = forward_returns.subtract(forward_returns.mean(axis=1), axis=0)
    elif adjust == "excess_over_benchmark":
        raise NotImplementedError(
            "Excess returns over benchmark is not implemented yet."
        )
    elif adjust == "vol_adjusted":
        forward_returns = forward_returns.divide(
            forward_returns.rolling(20).std(), axis=0
        )
    elif adjust == "vol_liquid_adjusted":
        raise NotImplementedError(
            "Volume-weighted return adjusted for stock size is not implemented yet."
        )
    elif adjust == "size_adjusted":
        raise NotImplementedError("Size-adjusted return is not implemented yet.")

    return forward_returns
