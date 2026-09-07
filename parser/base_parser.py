import pandas as pd
from abc import ABC, abstractmethod


COMMON_LOG_SCHEMA = [
    "date",
    "time",
    "pid",
    "log_level",
    "component",
    "event_type",
    "block_id",
    "source_ip",
    "destination_ip",
    "block_size",
    "raw_message",
]


class BaseParser(ABC):
    schema = COMMON_LOG_SCHEMA

    @abstractmethod
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parses a log file and returns a pandas DataFrame.
        """
        pass

    def empty_frame(self) -> pd.DataFrame:
        """
        Returns an empty DataFrame with the common parser schema.
        """
        return pd.DataFrame(columns=self.schema)
