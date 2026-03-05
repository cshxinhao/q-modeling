from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np


class Evaluator:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the Evaluator with a dataframe.
        Expected columns: [datetime, symbol, pred, 1d, 5d, 10d, industry, total_market_cap, adv, board]
        """
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["datetime"]):
            self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        # Ensure sorted for rolling/shifting operations
        self.df = self.df.sort_values(["symbol", "datetime"])

        # Cache for calculated results
        self.results = {}

    def calculate_coverage(self):
        """
        1. The coverage (number of stocks with prediction) per datetime.
        """
        cache_key = "coverage"
        if cache_key in self.results:
            return self.results[cache_key]

        coverage = self.df.groupby("datetime")["symbol"].count()
        self.results[cache_key] = coverage
        return coverage

    def calculate_autocorrelation(self, horizons=(1, 5, 10, 20)):
        """
        Calculate the auto-correlation of prediction.
        """
        cache_key = f"autocorr_{horizons}"
        if cache_key in self.results:
            return self.results[cache_key]

        results = []

        # We need to shift within each symbol
        # Since self.df is sorted by symbol, datetime, we can group and shift
        grouped = self.df.groupby("symbol")["pred"]

        for h in horizons:
            # Shift the predictions
            lagged_pred = grouped.shift(h)

            # Create a temporary dataframe for correlation calculation
            temp_df = pd.DataFrame(
                {
                    "datetime": self.df["datetime"],
                    "pred": self.df["pred"],
                    "pred_lag": lagged_pred,
                }
            )

            # Drop NaNs created by shifting
            temp_df = temp_df.dropna()

            if temp_df.empty:
                continue

            # Calculate correlation per datetime
            corr_series = temp_df.groupby("datetime").apply(
                lambda x: x["pred"].corr(x["pred_lag"])
            )
            corr_series.name = f"{h}d"
            results.append(corr_series)

        if not results:
            return pd.DataFrame()

        autocorr_df = pd.concat(results, axis=1)
        self.results[cache_key] = autocorr_df
        return autocorr_df

    def calculate_ic(self, horizons=("1d", "5d", "10d"), method="pearson"):
        """
        Calculate IC/RankIC.
        """
        cache_key = f"ic_{method}_{horizons}"
        if cache_key in self.results:
            return self.results[cache_key]

        results = []
        valid_horizons = [h for h in horizons if h in self.df.columns]

        if not valid_horizons:
            return pd.DataFrame()

        for h in valid_horizons:
            # Group by datetime and calculate correlation
            # We filter out NaNs in the target column
            temp_df = self.df[["datetime", "pred", h]].dropna()

            if temp_df.empty:
                continue

            corr_series = temp_df.groupby("datetime").apply(
                lambda x: x["pred"].corr(x[h], method=method)
            )
            corr_series.name = h
            results.append(corr_series)

        if not results:
            return pd.DataFrame()

        ic_df = pd.concat(results, axis=1)
        self.results[cache_key] = ic_df
        return ic_df

    def calculate_ic_by_market_cap(
        self, horizons=("1d", "5d", "10d"), groups=5, method="pearson"
    ):
        """
        Calculate IC/RankIC by market cap groups.
        """
        cache_key = f"ic_by_mcap_{method}_{horizons}_{groups}"
        if cache_key in self.results:
            return self.results[cache_key]

        results = {}
        valid_horizons = [h for h in horizons if h in self.df.columns]

        if not valid_horizons or "total_market_cap" not in self.df.columns:
            return pd.DataFrame()

        for h in valid_horizons:
            temp_df = self.df[["datetime", "pred", h, "total_market_cap"]].dropna()

            if temp_df.empty:
                continue

            def calc_group_ic(df_day):
                if len(df_day) < groups:
                    return pd.Series(index=range(1, groups + 1), dtype=float)

                ranks = df_day["total_market_cap"].rank(method="first", pct=True)
                groups_series = np.ceil(ranks * groups).astype(int)

                res = df_day.groupby(groups_series).apply(
                    lambda x: x["pred"].corr(x[h], method=method)
                )
                return res.reindex(range(1, groups + 1))

            ic_by_group = temp_df.groupby("datetime").apply(calc_group_ic)
            # Ensure columns are named properly
            ic_by_group.columns = [f"Group_{c}" for c in ic_by_group.columns]
            results[h] = ic_by_group

        if not results:
            return pd.DataFrame()

        final_df = pd.concat(results.values(), axis=1, keys=results.keys())
        self.results[cache_key] = final_df
        return final_df

    def calculate_grouped_returns(
        self,
        groups=10,
        horizon="1d",
        excess=False,
    ):
        cache_key = f"grouped_returns_{groups}_{horizon}_{excess}"
        if cache_key in self.results:
            return self.results[cache_key]

        if horizon not in self.df.columns:
            return pd.DataFrame()

        temp_df = self.df[["datetime", "pred", horizon]].dropna()

        if temp_df.empty:
            return pd.DataFrame()

        # # 1. Assign groups per datetime
        temp_df["group"] = np.ceil(
            temp_df.groupby("datetime")["pred"]
            .rank(pct=True, method="first")
            .mul(groups)
        ).astype(int)

        # 2. Calculate average return per group per datetime
        grouped_returns = (
            temp_df.groupby(["datetime", "group"])[horizon].mean().unstack()
        )

        # 3. If excess returns, subtract the daily mean
        if excess:
            # Daily equal-weighted return (market return)
            market_ret = temp_df.groupby("datetime")[horizon].mean()
            # Subtract market return from each column
            grouped_returns = grouped_returns.sub(market_ret, axis=0)

        self.results[cache_key] = grouped_returns
        return grouped_returns

    def calculate_grouped_turnover(
        self, groups=10, trade_cost_bps=25, periods_per_year=252
    ):
        """
        Calculate annualized turnover and trading cost for each group (equal weighted).
        Returns a table with index=[Turnover, Trading Cost] and columns=[Groups].
        """
        cache_key = f"grouped_turnover_{groups}_{trade_cost_bps}"
        if cache_key in self.results:
            return self.results[cache_key]

        temp_df = self.df[["datetime", "symbol", "pred"]].dropna()

        if temp_df.empty:
            return pd.DataFrame()

        # Assign groups
        temp_df["group"] = np.ceil(
            temp_df.groupby("datetime")["pred"]
            .rank(pct=True, method="first")
            .mul(groups)
        ).astype(int)

        # Equal weights within group
        # weight = 1 / count per group per datetime
        counts = temp_df.groupby(["datetime", "group"])["symbol"].transform("count")
        temp_df["weight"] = 1.0 / counts

        # Calculate stats
        # position_stats is defined at module level
        _, annualized_turnover, _, annualized_cost = position_stats(
            temp_df,
            periods_per_year=periods_per_year,
            trade_cost_bps=trade_cost_bps,
        )

        # Construct result table
        # annualized_turnover and annualized_cost are Series indexed by group
        res_df = pd.DataFrame(
            {
                "Turnover": annualized_turnover,
                "Trading Cost": annualized_cost,
            }
        ).T

        # Sort columns (groups)
        res_df = res_df.sort_index(axis=1)

        self.results[cache_key] = res_df
        return res_df


def _returns_stats(
    returns: Union[pd.Series, pd.DataFrame],
    periods_per_year: int = 252,
    compound: bool = False,
) -> pd.Series:
    """
    Calculate basic statistics for returns.

    Parameters:
    - returns: pd.Series, time series of returns.
    - periods_per_year: int, number of periods per year.
    - compound: bool, whether to calculate compound returns.

    Returns:
    - pd.Series, basic statistics for returns.
    """
    stats_dict = {}

    returns = returns.dropna()

    if returns.empty:
        return pd.Series()

    # Valid data points
    stats_dict["Start Date"] = returns.index[0].strftime("%Y-%m-%d")
    stats_dict["End Date"] = returns.index[-1].strftime("%Y-%m-%d")

    # Compound returns
    if compound:
        nav = (returns + 1).cumprod()
        stats_dict["Annual Returns"] = nav.iloc[-1] ** (periods_per_year / len(nav)) - 1
        stats_dict["Best Month"] = (
            returns.add(1).groupby(returns.index.month).prod().sub(1).max()
        )
        stats_dict["Worst Month"] = (
            returns.add(1).groupby(returns.index.month).prod().sub(1).min()
        )
    else:
        nav = returns.cumsum() + 1
        stats_dict["Annual Returns"] = returns.mean() * periods_per_year
        stats_dict["Best Month"] = returns.groupby(returns.index.month).sum().max()
        stats_dict["Worst Month"] = returns.groupby(returns.index.month).sum().min()

    stats_dict["Total Returns"] = nav.iloc[-1] - 1
    stats_dict["Annual Volatility"] = returns.std() * np.sqrt(periods_per_year)
    stats_dict["Sharpe Ratio"] = (
        stats_dict["Annual Returns"] / stats_dict["Annual Volatility"]
        if stats_dict["Annual Volatility"] != 0
        else np.nan
    )
    stats_dict["Max Drawdown"] = nav.div(nav.cummax()).sub(1).min()

    # Win ratio
    stats_dict["Win Ratio"] = (returns > 0).mean()
    stats_dict["Win Month"] = (returns.groupby(returns.index.month).mean() > 0).mean()

    # Prettify the stats
    pct_keys = [
        "Annual Returns",
        "Annual Volatility",
        "Max Drawdown",
        "Win Ratio",
        "Win Month",
        "Best Month",
        "Total Returns",
        "Worst Month",
    ]
    for k in pct_keys:
        if k in stats_dict:
            stats_dict[k] = f"{stats_dict[k]:.2%}"

    float_keys = ["Sharpe Ratio"]
    for k in float_keys:
        if k in stats_dict and not pd.isna(stats_dict[k]):
            stats_dict[k] = round(stats_dict[k], 2)

    keys = [
        "Start Date",
        "End Date",
        "Annual Returns",
        "Annual Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Win Ratio",
        "Win Month",
        "Best Month",
        "Total Returns",
        "Worst Month",
    ]
    return pd.Series(stats_dict).reindex(keys)


def returns_stats(
    returns: Union[pd.DataFrame, pd.Series],
    periods_per_year: int = 252,
    compound: bool = False,
):
    """
    Calculate basic statistics for returns.
    """

    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    stats_dict = {}
    for col in returns.columns:
        stats_dict[col] = _returns_stats(returns[col], periods_per_year, compound)

    return pd.DataFrame(stats_dict)


def position_stats(
    position_df: pd.DataFrame,
    periods_per_year: int = 252,
    trade_cost_bps: int = 25,
) -> tuple[
    Union[pd.Series, pd.DataFrame],
    Union[float, pd.Series],
    Union[pd.Series, pd.DataFrame],
    Union[float, pd.Series],
]:
    """
    Calculate the turnover rate and transaction cost for each datetime.

    Turnover rate is defined as 1/2 * sum(|w_t - w_{t-1}|).
    Transaction cost is calculated as turnover * trade_cost_bps / 10000.
    The first date's turnover and cost are set to NaN.

    If 'group' column is present in position_df, calculations are performed per group.

    Parameters:
    - position_df: pd.DataFrame with columns ['datetime', 'symbol', 'weight'] and optionally ['group']
    - periods_per_year: int, number of periods per year (default 252)
    - trade_cost_bps: int, two-way transaction cost in basis points (default 25)

    Returns:
    - turnover_series: pd.Series or pd.DataFrame, turnover rate for each datetime (per group if applicable)
    - annualized_turnover: float or pd.Series, average turnover * periods_per_year
    - cost_series: pd.Series or pd.DataFrame, transaction cost for each datetime (per group if applicable)
    - annualized_cost: float or pd.Series, average cost * periods_per_year
    """
    df = position_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    if "group" in df.columns:
        # Pivot to wide format: index=[datetime, group], columns=symbol, values=weight
        weights_wide = df.pivot(
            index=["datetime", "group"], columns="symbol", values="weight"
        ).fillna(0.0)
        weights_wide = weights_wide.sort_index()

        # Calculate diff per group
        # groupby(level='group') ensures we diff within the same group across time
        diff = weights_wide.groupby("group").diff()
    else:
        # Pivot to wide format: index=datetime, columns=symbol, values=weight
        weights_wide = df.pivot(
            index="datetime", columns="symbol", values="weight"
        ).fillna(0.0)
        weights_wide = weights_wide.sort_index()

        # Calculate turnover
        # diff() gives w_t - w_{t-1}
        diff = weights_wide.diff()

    # Calculate one-way turnover: 1/2 * sum(|diff|)
    turnover = diff.abs().sum(axis=1) * 0.5

    if "group" in df.columns:
        # Unstack to get datetime as index and groups as columns
        turnover = turnover.unstack("group")

    # Set the first turnover to NaN
    turnover.iloc[0] = np.nan

    annualized_turnover = turnover.mean() * periods_per_year

    # Calculate transaction cost
    # Cost = Turnover * trade_cost_bps / 10000
    cost = turnover * trade_cost_bps / 10000.0
    annualized_cost = cost.mean() * periods_per_year

    return turnover, annualized_turnover, cost, annualized_cost
