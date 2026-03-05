# Key component of defining the model pipeline
# Execute and coordinate several elements
# Responsible includes:
# 1. Load the raw data and features
# 2. Build labels
# 3. Feature engineering
# 4. Train the model: include cross-validation for hyperparameter tuning
# 5. Evaluate the model
# 6. Save the model

import logging
import pandas as pd

from src.data_ingest import RawDataLoader, FeatureLoader
from src.feature import (
    FeatureCleaner,
    FeatureDiscretizer,
    FeatureDimensionReducer,
    FeatureDeriver,
)
from src.label import build_labels
from src.model import train_model, save_model, load_model, predict
from src.evaluator import Evaluator
from src.logger import setup_logger

logger = setup_logger(__name__, level=logging.DEBUG)


def simple_pipeline(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
):

    # --------------------------------------------------------------------------------
    # Training

    # Load features
    feature_loader = FeatureLoader(train_start, train_end)
    features = feature_loader.load_features(
        [
            "10-day avg excess returns",
            "10-day avg intraday returns",
            "10-day avg negative returns",
            "10-day avg overnight returns",
            "10-day avg positive returns",
            "10-day avg residual returns",
            "10-day avg returns",
            "10-day avg volume weighted returns v2",
            "10-day compound returns",
            "10-day max returns",
        ]
    )
    # Normalize features
    features = FeatureCleaner.normalize_features(features)

    # Load the raw data
    raw_data_loader = RawDataLoader(train_start, train_end)
    raw_data = raw_data_loader.load_fields(
        ["close", "vwap", "volume", "amount", "cap_ff"]
    )

    # Build labels

    # --------------------------------------------------------------------------------
    # Predicting


def cv_pipeline():
    pass
