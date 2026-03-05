# Given the features, perform feature engineering
# Potential engineering elements:
# - feature selection
# - orthogonalization: extract incremental information out of features
# - feature scaling: normalize features to have zero mean and unit variance
# - feature encoding: convert categorical features to numerical features
# - feature interaction: create new features by combining existing features
# - feature transformation: apply mathematical transformations to features
# - feature binning: discretize continuous features into bins

import pandas as pd


class FeatureCleaner:
    def encode_categorical_features():
        pass

    def normalize_features(df: pd.DataFrame):
        """
        Normalize features.

        Parameters:
        df (pd.DataFrame): DataFrame containing features with ['datetime', 'symbol'] index.

        Returns:
        pd.DataFrame: DataFrame with normalized features.
        """
        if df.empty:
            return df
        mu = df.groupby("datetime").transform("mean")
        sigma = df.groupby("datetime").transform("std")
        sigma = sigma.where(sigma != 0)
        df = df.fillna(mu)
        df = df.sub(mu).div(sigma)
        return df


class FeatureDiscretizer:
    """
    The financial signal can be noisy somtimes.
    Binning can help to smooth out the signal and make it more stable.
    """

    def bin_features():
        """

        Method 1: equal width binning. Divide the range into N intervals of the same size.
        * pros:
            - simple and easy to implement
            - equal size bins
            - preserve the range structure
        * cons:
            - sensitive to outliers
            - can result in empty bins
        * best use base: uniformly distributed data

        Method 2: equal frequency binning. Divide the samples into N bins with equal quantity of samples.
        * pros:
            - handle skewed data well
            - spread out crowded values
            - suitable for features with skewed distributions, can flatten the distribution
        * cons:
            - may assign identical values to different bins
        * best use case: skewed data (e.g. stock returns, income, wealth)

        """

        pass

    def rank_features(pct=True):
        """
        best use case: non-parametric models
        """

        pass

    def binarize_features(threshold: float = 0.0):
        pass

    def cluster_features():
        pass

    def tree_split_features():
        """
        Uses a tree to find splits that maximize the information gain or Gini.
        Pros:
            - high predictive power
            - bins are optimized for the target
            - recursively partition the feature space into discrete, non-overlapping regions
        Cons:
            - risk of overfitting
            - requires a target
            - incapable of extrapolating trends beyong the observed range
            - out-of-distribution data
            - be careful not to snoop the data
        """
        pass


class FeatureDimensionReducer:
    def select_features():
        pass

    def orthogonalize_features():
        pass


class FeatureDeriver:
    def derive_feature_interactions():
        pass
