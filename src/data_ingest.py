import os
import pandas as pd
import pyarrow.parquet as pq
import warnings
from pathlib import Path
from src.logger import setup_logger
import logging

logger = setup_logger(__name__, level=logging.DEBUG)


class RawDataLoader:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self.start = start
        self.end = end
        self.directory = Path(os.getenv("RAW_DATA_DIR"))

    def load_fields(self, fields: list[str] = None) -> pd.DataFrame:
        filters = [
            ("datetime", ">=", self.start),
            ("datetime", "<=", self.end),
        ]
        try:
            if fields is None:
                df = pd.read_parquet(
                    self.directory,
                    filters=filters,
                )
            else:
                df = pd.read_parquet(
                    self.directory,
                    filters=filters,
                    columns=["datetime", "symbol"] + fields,
                )
        except Exception:
            warnings.warn(f"Failed to load {fields}, skip.")
            return pd.DataFrame()
        df = df.set_index(["datetime", "symbol"]).sort_index()
        return df


class FeatureLoader:
    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self.start = start
        self.end = end
        self.directory = Path(os.getenv("FEATURE_DATA_DIR"))
        self._find_name_to_filename()

    def _find_name_to_filename(self):
        self.name_to_filename = {}
        for filename in self.directory.glob("*.parquet"):
            meta = pq.read_metadata(filename)
            name = meta.schema.names[-1]
            self.name_to_filename[name] = filename
        return self.name_to_filename

    def load_feature(self, name: str) -> pd.DataFrame:

        filters = [
            ("datetime", ">=", self.start),
            ("datetime", "<=", self.end),
        ]
        try:
            filename = self.name_to_filename[name]
            df = pd.read_parquet(
                filename,
                filters=filters,
            )
        except Exception:
            warnings.warn(f"Failed to load {name}, skip.")
            return pd.DataFrame()
        df = df.set_index(["datetime", "symbol"]).sort_index()

        return df

    def load_features(self, names: list[str] = None) -> pd.DataFrame:

        if names is None:
            names = list(self.name_to_filename.keys())

        logger.debug(f"Loading {len(names)} features")

        container = []
        for name in names:
            df = self.load_feature(name)
            container.append(df)

        if not container:
            return pd.DataFrame()

        return pd.concat(container, axis=1).sort_index()
