# The scheduler is a wrapper of the pipeline
# If basically controls the training, predicting, persisting results on a daily basis
# In most cases, the model is retrained every N weeks, and the trained model is persisted

import pandas as pd
from src.pipeline import simple_pipeline


def simple_window_scheduler(
    start_year: int,
    end_year: int,
    retrain_month: int,
    window_type: str,
    window_size: int,  # in months
    pipeline: callable,
):
    """
    if window_type = rolling, window_size is the rolling window size
    if window_type = expanding, window_size is the first window size
    """

    assert window_type in ["rolling", "expanding"], (
        "window_type must be either 'rolling' or 'expanding'"
    )

    train_start = pd.Timestamp(f"{start_year}-01-01")
    train_end = train_start + pd.offsets.MonthEnd(window_size)
    predict_start = train_end + pd.offsets.Day(1)
    predict_end = predict_start + pd.offsets.MonthEnd(retrain_month)

    while predict_end.year <= end_year:
        pipeline(train_start, train_end, predict_start, predict_end)

        # Update the training period
        if window_type == "rolling":
            train_start = train_start + pd.offsets.MonthBegin(retrain_month)
        train_end = train_end + pd.offsets.MonthEnd(retrain_month)
        predict_start = train_end + pd.offsets.Day(1)
        predict_end = predict_start + pd.offsets.MonthEnd(retrain_month)
