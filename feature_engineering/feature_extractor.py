from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_OUTPUT_PATH = Path("data/features/features.csv")
REQUIRED_CLEAN_COLUMNS = [
    "datetime",
    "log_level",
    "component",
    "event_type",
    "block_id",
    "source_ip",
    "destination_ip",
    "block_size",
]
FEATURE_COLUMNS = [
    "hour_of_day",
    "is_error",
    "component_encoded",
    "event_type_encoded",
    "block_size",
    "event_frequency",
    "source_ip_encoded",
    "destination_ip_encoded",
]


class HDFSFeatureExtractor:
    """
    Converts cleaned HDFS logs into ML-ready numerical features.
    """

    def extract(
        self,
        logs: pd.DataFrame | str | Path,
        output_path: str | Path = DEFAULT_FEATURE_OUTPUT_PATH,
    ) -> pd.DataFrame:
        frame = self._load_frame(logs)
        self.validate_schema(frame)

        prepared = frame.copy()
        prepared["datetime"] = pd.to_datetime(prepared["datetime"], errors="coerce")
        prepared["block_size"] = pd.to_numeric(prepared["block_size"], errors="coerce").fillna(0)
        prepared = prepared.dropna(subset=["datetime"]).copy()
        prepared = prepared.sort_values("datetime").reset_index(drop=True)

        for column in ["component", "event_type", "source_ip", "destination_ip", "block_id"]:
            prepared[column] = prepared[column].astype("string").str.strip().fillna("UNKNOWN")
            prepared[column] = prepared[column].replace({"": "UNKNOWN", "<NA>": "UNKNOWN"})

        event = pd.DataFrame(index=prepared.index)
        event["block_id"] = prepared["block_id"]
        event["hour_of_day"] = prepared["datetime"].dt.hour.astype("int64")
        event["is_error"] = (
            prepared["log_level"].astype("string").str.upper().eq("ERROR").astype("int64")
        )
        event["component_encoded"] = self._encode_categories(prepared["component"])
        event["event_type_encoded"] = self._encode_categories(prepared["event_type"])
        event["block_size"] = prepared["block_size"].astype("float64")
        event["source_ip_encoded"] = self._encode_categories(prepared["source_ip"])
        event["destination_ip_encoded"] = self._encode_categories(prepared["destination_ip"])

        grouped = event.groupby("block_id", sort=False)
        aggregated = grouped.agg(
            hour_of_day=("hour_of_day", "first"),
            is_error=("is_error", "max"),
            block_size=("block_size", "max"),
            event_frequency=("block_id", "size"),
        ).reset_index()

        for column in [
            "component_encoded",
            "event_type_encoded",
            "source_ip_encoded",
            "destination_ip_encoded",
        ]:
            dominant = self._dominant_value_per_group(event, "block_id", column)
            aggregated[column] = aggregated["block_id"].map(dominant)

        aggregated["hour_of_day"] = aggregated["hour_of_day"].astype("int64")
        aggregated["is_error"] = aggregated["is_error"].astype("int64")
        aggregated["block_size"] = aggregated["block_size"].astype("float64")
        aggregated["event_frequency"] = aggregated["event_frequency"].astype("int64")

        block_id = aggregated["block_id"].reset_index(drop=True)
        features = aggregated[FEATURE_COLUMNS].reset_index(drop=True)
        result = pd.concat([block_id, features], axis=1)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        return features

    def validate_schema(self, frame: pd.DataFrame) -> None:
        missing_columns = [
            column for column in REQUIRED_CLEAN_COLUMNS if column not in frame.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing required cleaned columns: {missing_columns}")

    def _load_frame(self, logs: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(logs, pd.DataFrame):
            return logs
        return pd.read_csv(logs)

    @staticmethod
    def _encode_categories(series: pd.Series) -> pd.Series:
        values = series.astype("string").fillna("UNKNOWN")
        categories = {value: index for index, value in enumerate(sorted(values.unique()))}
        return values.map(categories).astype("int64")

    @staticmethod
    def _dominant_value_per_group(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
        """
        Returns the most frequent value_col per group_col as a Series indexed
        by group_col, breaking ties toward the smallest value_col.
        """
        counts = (
            frame.groupby([group_col, value_col], sort=False)
            .size()
            .reset_index(name="_count")
        )
        counts = counts.sort_values(["_count", value_col], ascending=[False, True], kind="mergesort")
        counts = counts.drop_duplicates(subset=[group_col], keep="first")
        return counts.set_index(group_col)[value_col]


def extract_hdfs_features(
    input_path: str | Path = "data/processed/hdfs_cleaned.csv",
    output_path: str | Path = DEFAULT_FEATURE_OUTPUT_PATH,
) -> pd.DataFrame:
    return HDFSFeatureExtractor().extract(input_path, output_path)


if __name__ == "__main__":
    extract_hdfs_features()
