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
        # "device": "cuda",
    }
    LABEL_HORIZON = 5
    MODEL_SAVE_DIR = MODEL_SAVE_DIR / "xgb5d_20260310"

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
