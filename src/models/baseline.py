import pandas as pd
import joblib

from src.utils import (
    get_model_id,
    get_model_files,
    calc_forward_returns,
    save_meta,
    load_meta,
)
from src.data_ingest import FeatureLoader, RawDataLoader
from src.feature_enginner import FeatureCleaner
from src.logger import setup_logger
from src.interface import ModelInterface


logger = setup_logger(__name__)


class BaselineRegModel(ModelInterface):
    """
    Baseline Regression Model Spec:
    * Feature engineering: Normalise the features. Let the model decide the weighting of all features.
    * Label engineering: Single label.  such as 5d, 10d, 20d, etc.
    * Model: Single model, such as xgb, lgb, lasso, ridge, etc.
    * Optimisation: Fixed hyperparameters.
    """

    def __init__(
        self,
        base_model_name: str,
        model_params: dict,
        label_horizon: int,
        cupy: bool = False,
        save_dir: str = "model_dir",
        train_start: pd.Timestamp = None,
        train_end: pd.Timestamp = None,
        test_start: pd.Timestamp = None,
        test_end: pd.Timestamp = None,
    ):
        # Import necessary modules
        if base_model_name == "xgb":
            from xgboost import XGBRegressor as ModelClass
        elif base_model_name == "lgb":
            from lightgbm import LGBMRegressor as ModelClass
        else:
            raise ValueError(f"base_model_name {base_model_name} is not supported")

        # Set the base model
        self.base_model_name = base_model_name
        self.ModelClass = ModelClass
        self.model_params = model_params
        self.label_horizon = label_horizon
        self.cupy = cupy
        self.save_dir = save_dir

        # Set the training and testing period
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        # Initialization
        self.fitted = False
        if train_start and train_end:
            self._init_model_id()

    def _init_model_id(self):
        self.model_id = get_model_id(
            base_model_name=self.base_model_name,
            model_params=self.model_params,
            label=f"label_{self.label_horizon}d",
            train_start=self.train_start.strftime("%Y%m%d"),
            train_end=self.train_end.strftime("%Y%m%d"),
        )
        self.model_files = get_model_files(
            model_id=self.model_id,
            save_dir=self.save_dir,
        )

    def _get_X(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Get the features for training and testing.
        """

        # Load features
        feature_loader = FeatureLoader(start=start, end=end)
        features = feature_loader.load_features(
            # names=[
            #     "10-day avg excess returns",
            #     "10-day avg intraday returns",
            #     "10-day avg negative returns",
            #     "10-day avg overnight returns",
            #     "10-day avg positive returns",
            #     "10-day avg residual returns",
            #     "10-day avg returns",
            #     "10-day avg volume weighted returns v2",
            #     "10-day compound returns",
            #     "10-day max returns",
            # ]
        )

        # Normalize features
        features = FeatureCleaner.normalize_features(features)

        # No additional feature engineering

        return features

    def _get_y(self, start: pd.Timestamp, end: pd.Timestamp):
        """
        Get the labels for training and testing.
        """

        # Load raw data
        raw_data_loader = RawDataLoader(start=start, end=end)
        raw_data = raw_data_loader.load_fields(
            ["close", "vwap", "volume", "amount", "cap_ff"]
        )

        # Calculate forward returns
        buy_price = raw_data["vwap"].unstack()
        sell_price = raw_data["vwap"].unstack()
        forward_returns = (
            calc_forward_returns(
                buy_price=buy_price,
                sell_price=sell_price,
                horizon=self.label_horizon,
                delay=1,
                adjust="excess_over_eqw",
            )
            .stack()
            .dropna()
        )

        # Calculate labels - remove outliers
        q1 = forward_returns.groupby("datetime").transform("quantile", q=0.25)
        q3 = forward_returns.groupby("datetime").transform("quantile", q=0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        is_outliers = (forward_returns < lower) | (forward_returns > upper)
        forward_returns = forward_returns.where(~is_outliers)
        forward_returns = forward_returns.dropna()

        # Calculate labels - normalise
        mu = forward_returns.groupby("datetime").transform("mean")
        sigma = forward_returns.groupby("datetime").transform("std")
        labels = forward_returns.sub(mu).div(sigma)

        labels.columns = f"label_{self.label_horizon}d"

        return labels

    def _get_model(self):

        # If the model fitted, load it
        model_file = self.model_files[0]
        if model_file.exists():
            logger.info(f"Model already trained, load it from {model_file}")
            model = joblib.load(model_file)
            self.fitted = True
            return model

        # If the model not fitted, initialize it
        logger.info("Model not trained, initialize it")
        model = self.ModelClass(**self.model_params)
        return model

    def train(self):

        logger.info(
            f"{self.model_id}: Training model for {self.train_start.strftime('%Y%m%d')} to {self.train_end.strftime('%Y%m%d')}"
        )

        X = self._get_X(start=self.train_start, end=self.train_end)
        idx, cols = X.index, X.columns
        y = self._get_y(start=self.train_start, end=self.train_end)
        X, y = X.align(y, join="inner", axis=0)
        model = self._get_model()

        model.fit(X, y)
        self.fitted = True

        # Save the model
        model_file = self.model_files[0]
        joblib.dump(model, model_file)

        # Save meta
        meta_file = self.model_files[1]
        save_meta(
            train_start=self.train_start,
            train_end=self.train_end,
            label=f"label_{self.label_horizon}d",
            model_params=self.model_params,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            selected_features=cols.to_list(),
            meta_filename=meta_file,
        )

        return self

    def predict(self):

        logger.info(
            f"{self.model_id}: Predicting for {self.test_start.strftime('%Y%m%d')} to {self.test_end.strftime('%Y%m%d')}"
        )

        if not self.fitted:
            raise ValueError("Model not fitted. Please train the model first.")
        model = self._get_model()
        meta_file = self.model_files[1]
        meta = load_meta(meta_file)
        selected_features = meta["selected_features"]
        X = self._get_X(start=self.test_start, end=self.test_end)[selected_features]
        idx, cols = X.index, X.columns

        y_pred = model.predict(X)

        # Save the predictions
        pred_path = self.model_files[2]
        y_pred = pd.DataFrame(y_pred, index=idx, columns=["pred"])
        y_pred.to_parquet(pred_path)

        return y_pred

    def refresh(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.fitted = False
        self._init_model_id()
