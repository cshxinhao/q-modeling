from pathlib import Path
from src.models.baseline import BaselineRegModel
from src.scheduler import simple_window_scheduler
from src.settings import MODEL_SAVE_DIR

if __name__ == "__main__":
    # TODO: Sample Config, needs to be formatted in config file in a more elegant way
    START_YEAR = 2012
    END_YEAR = 2026
    RETRAIN_MONTH = 6
    WINDOW_TYPE = "rolling"
    WINDOW_SIZE = 24

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
    MODEL_SAVE_DIR = Path(MODEL_SAVE_DIR) / "baseline_xgb5d"

    model = BaselineRegModel(
        base_model_name=BASE_MODEL_NAME,
        model_params=MODEL_PARAMS,
        label_horizon=LABEL_HORIZON,
        save_dir=MODEL_SAVE_DIR,
        cupy=False,
    )
    simple_window_scheduler(
        start_year=START_YEAR,
        end_year=END_YEAR,
        retrain_month=RETRAIN_MONTH,
        window_type=WINDOW_TYPE,
        window_size=WINDOW_SIZE,
        model=model,
    )

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
    oof_preds.to_parquet(MODEL_SAVE_DIR / "oof_preds.parquet")
