from pathlib import Path

import pandas as pd

from parser.base_parser import COMMON_LOG_SCHEMA


DEFAULT_CLEANED_OUTPUT_PATH = Path("data/processed/hdfs_cleaned.csv")
REQUIRED_FIELDS = [
    "date",
    "time",
    "pid",
    "log_level",
    "component",
    "event_type",
    "block_id",
    "raw_message",
]
OPTIONAL_CATEGORICAL_FIELDS = ["source_ip", "destination_ip"]


class HDFSDataCleaner:
    """
    Validates and cleans structured HDFS logs for feature engineering.
    """

    schema = COMMON_LOG_SCHEMA

    def clean(
        self,
        logs: pd.DataFrame | str | Path,
        output_path: str | Path = DEFAULT_CLEANED_OUTPUT_PATH,
    ) -> pd.DataFrame:
        frame = self._load_frame(logs)
        self.validate_schema(frame)

        cleaned = frame.copy()
        cleaned = cleaned.drop_duplicates()
        cleaned = self._coerce_column_types(cleaned)
        cleaned = self._drop_invalid_required_rows(cleaned)
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(output, index=False)
        return cleaned

    def validate_schema(self, frame: pd.DataFrame) -> None:
        missing_columns = [column for column in self.schema if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required schema columns: {missing_columns}")

    def _load_frame(self, logs: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(logs, pd.DataFrame):
            return logs
        return pd.read_csv(logs)

    def _coerce_column_types(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()

        cleaned["date"] = cleaned["date"].astype("string").str.strip()
        cleaned["time"] = cleaned["time"].astype("string").str.strip()
        cleaned["datetime"] = pd.to_datetime(
            cleaned["date"] + " " + cleaned["time"],
            errors="coerce",
        )

        cleaned["pid"] = pd.to_numeric(cleaned["pid"], errors="coerce")
        cleaned["block_size"] = pd.to_numeric(cleaned["block_size"], errors="coerce").fillna(0)

        for column in ["log_level", "component", "event_type", "block_id", "raw_message"]:
            cleaned[column] = cleaned[column].astype("string").str.strip()

        for column in OPTIONAL_CATEGORICAL_FIELDS:
            cleaned[column] = cleaned[column].astype("string").str.strip().fillna("UNKNOWN")
            cleaned[column] = cleaned[column].replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})

        return cleaned

    def _drop_invalid_required_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.dropna(subset=["datetime", "pid"]).copy()

        for column in ["log_level", "component", "event_type", "block_id", "raw_message"]:
            cleaned = cleaned[
                cleaned[column].notna()
                & (cleaned[column] != "")
                & (cleaned[column] != "<NA>")
            ]

        cleaned["pid"] = cleaned["pid"].astype("int64")
        cleaned["block_size"] = cleaned["block_size"].astype("float64")

        return cleaned


def clean_hdfs_logs(
    input_path: str | Path = "data/processed/hdfs_structured.csv",
    output_path: str | Path = DEFAULT_CLEANED_OUTPUT_PATH,
) -> pd.DataFrame:
    return HDFSDataCleaner().clean(input_path, output_path)


if __name__ == "__main__":
    clean_hdfs_logs()
