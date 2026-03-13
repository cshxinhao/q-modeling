from pathlib import Path
import pandas as pd
from src.models.baseline import BaselineRegModel
from src.scheduler import simple_window_scheduler
from src.settings import MODEL_SAVE_DIR

if __name__ == "__main__":
    BASE_MODEL_NAME = "xgb"
    MODEL_PARAMS = {
        "n_estimators": 5800,
        "max_depth": 8,
        "learning_rate": 0.0001,
        "min_child_weight": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0.0,
        "lambda": 1.0,
        "gamma": 0.0,
        "device": "cuda",
    }
    LABEL_HORIZON = 5
    MODEL_SAVE_DIR = Path(MODEL_SAVE_DIR) / "baseline_xgb5d_live"

    train_start = pd.Timestamp("2020-01-01")
    train_end = pd.Timestamp("2025-12-31")
    test_start = pd.Timestamp("2026-01-01")
    test_end = pd.Timestamp.now() + pd.offsets.MonthEnd(1)
    model = BaselineRegModel(
        base_model_name=BASE_MODEL_NAME,
        model_params=MODEL_PARAMS,
        label_horizon=LABEL_HORIZON,
        save_dir=MODEL_SAVE_DIR,
        cupy=False,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
    )
    model.train()
    model.predict(replace=True)

    # Manual aggregate all oof predictions
    import pandas as pd

    oof_files = []
    for model_file in MODEL_SAVE_DIR.glob("pred*.parquet"):
        oof_files.append(model_file)
    oof_preds = pd.read_parquet(oof_files)
    add_info = pd.read_parquet(
        r"D:\data_warehouse\clean_data\ml_common_data",
        columns=[
            "datetime",
            "symbol",
            "vwap_fr1d_delay1d",
            "vwap_fr5d_delay1d",
            "vwap_fr10d_delay1d",
            "industry",
            "cap_total",
            "adv",
            "board",
        ],
    ).rename(
        columns={
            "vwap_fr1d_delay1d": "1d",
            "vwap_fr5d_delay1d": "5d",
            "vwap_fr10d_delay1d": "10d",
        }
    )
    oof_preds = oof_preds.merge(add_info, on=["datetime", "symbol"], how="left")
    oof_preds.to_parquet(MODEL_SAVE_DIR / "oof_preds.parquet", index=False)

    # Select Top N
    df = pd.read_parquet(
        r"D:\model_data_warehouse\china_all\baseline_xgb5d_live\oof_preds.parquet",
        filters=[("datetime", ">=", test_start)],
    )
    df = df.dropna(subset=["adv", "cap_total", "board"])

    SEL_N = 15
    select_df = (
        df.query('board == "MAIN" and adv >= 10e6')
        .sort_values("pred")
        .groupby("datetime")
        .tail(SEL_N)
    )
    select_df = select_df.sort_values(["datetime", "symbol"])
    position = select_df.groupby("datetime")["symbol"].apply(set)
    turnover_rate = position.diff().dropna().apply(len).div(SEL_N)
    transaction_cost = turnover_rate * 15e-4
    pr = select_df.groupby("datetime")["1d"].mean()
    pr.cumsum().plot()
    prac = pr - transaction_cost
    prac.cumsum().plot()

    select_df.to_parquet(MODEL_SAVE_DIR / "oof_select.parquet", index=False)
