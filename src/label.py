import pandas as pd


def calc_forward_returns(
    buy_price: pd.DataFrame,
    sell_price: pd.DataFrame,
    horizon: int,
    delay: int,
    adjust: str = None,
):
    """
    buy_price: pd.DataFrame with index = 'datetime', column = 'symbol'
    sell_price: pd.DataFrame with index = 'datetime', column = 'symbol'
    The reason to distinguish buy_price and sell_price are:
    1. In markets with limit up/down rules, if the stock is limit up, you can't buy.
    2. The stamp duty is sometimes charged only on the sell side.
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


def build_labels(
    market_df: pd.DataFrame,
    horizons=("1d", "2d", "5d", "10d", "20d", "60d"),
    adjust: str = None,
):
    """
    Build labels for market data based on forward returns.

    Parameters:
    market_df (pd.DataFrame): DataFrame containing market data with ['datetime', 'symbol'] index and 'vwap' column.
    horizons (list[str], optional): List of horizons in days for which to calculate forward returns. Default is ("1d", "2d", "5d", "10d", "20d", "60d").

    Returns:
    pd.DataFrame: DataFrame with labels for each horizon.
    """
    if market_df.empty:
        return pd.DataFrame()

    # Detect limit up/down, mask it with NaN according to buy/sell action

    # Calculate forward returns
    buy_price = market_df["vwap"].unstack()
    sell_price = market_df["vwap"].unstack()
    forward_returns = pd.concat(
        [
            calc_forward_returns(
                buy_price, sell_price, int(horizon[:-1]), adjust=adjust
            )
            for horizon in horizons
        ],
        axis=1,
    )

    # Preprocess forward returns and build labels
    q1 = forward_returns.groupby("datetime").transform("quantile", q=0.25)
    q3 = forward_returns.groupby("datetime").transform("quantile", q=0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    is_outliers = (forward_returns < lower) | (forward_returns > upper)
    forward_returns = forward_returns.where(~is_outliers)
    mu = forward_returns.groupby("datetime").transform("mean")
    sigma = forward_returns.groupby("datetime").transform("std")
    labels = forward_returns.sub(mu).div(sigma)
    labels.columns = "label_" + labels.columns
    return labels, forward_returns
