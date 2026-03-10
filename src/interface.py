from abc import ABC, abstractmethod
import pandas as pd


class ModelInterface(ABC):
    """
    Interface for all models
    """

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def refresh(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ):
        pass
