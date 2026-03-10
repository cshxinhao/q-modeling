import pandas as pd
from src.data_ingest import FeatureLoader, RawDataLoader
from src.feature_enginner import FeatureCleaner
from src.utils import calc_forward_returns


def get_labels():

    horizons = (1, 2, 5, 10, 20, 60)

    # Load raw data
    raw_data_loader = RawDataLoader(
        start=pd.Timestamp("2012-01-01"), end=pd.Timestamp("2026-12-31")
    )
    raw_data = raw_data_loader.load_fields(
        ["close", "vwap", "volume", "amount", "cap_ff"]
    )

    # Calculate forward returns
    buy_price = raw_data["vwap"].unstack()
    sell_price = raw_data["vwap"].unstack()
    forward_returns = pd.concat(
        [
            calc_forward_returns(
                buy_price=buy_price,
                sell_price=sell_price,
                horizon=horizon,
                delay=1,
                adjust="excess_over_eqw",
            ).stack().dropna()
            for horizon in horizons
        ],
        axis=1,
    )

    # Calculate labels - remove outliers
    q1 = forward_returns.groupby("datetime").transform("quantile", q=0.25)
    q3 = forward_returns.groupby("datetime").transform("quantile", q=0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    is_outliers = (forward_returns < lower) | (forward_returns > upper)
    forward_returns = forward_returns.where(~is_outliers)
    forward_returns = forward_returns

    # Calculate labels - normalise
    mu = forward_returns.groupby("datetime").transform("mean")
    sigma = forward_returns.groupby("datetime").transform("std")
    labels = forward_returns.sub(mu).div(sigma)

    labels.columns = [f"label_{horizon}d" for horizon in horizons]

    return labels


def dump_labels():

    labels = get_labels()
    labels.to_parquet("./colab_projects/labels.parquet")


if __name__ == "__main__":
    dump_labels()
