import tempfile
import unittest
from pathlib import Path

import pandas as pd

from feature_engineering.data_cleaner import HDFSDataCleaner
from feature_engineering.feature_extractor import (
    FEATURE_COLUMNS,
    HDFSFeatureExtractor,
)


class HDFSDataCleanerTest(unittest.TestCase):
    def test_cleaner_validates_schema(self):
        frame = self._structured_logs().drop(columns=["event_type"])

        with self.assertRaises(ValueError):
            HDFSDataCleaner().clean(frame, self._temp_output_path("cleaned.csv"))

    def test_cleaner_removes_duplicates_and_coerces_types(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "hdfs_cleaned.csv"
            cleaned = HDFSDataCleaner().clean(self._structured_logs(), output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(len(cleaned), 2)
            self.assertFalse(cleaned[
                [
                    "date",
                    "time",
                    "pid",
                    "log_level",
                    "component",
                    "event_type",
                    "block_id",
                    "raw_message",
                ]
            ].isna().any().any())
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["datetime"]))
            self.assertTrue(pd.api.types.is_integer_dtype(cleaned["pid"]))
            self.assertTrue(pd.api.types.is_float_dtype(cleaned["block_size"]))
            self.assertEqual(cleaned.iloc[1]["block_size"], 0)
            self.assertEqual(cleaned.iloc[1]["source_ip"], "UNKNOWN")
            self.assertEqual(cleaned.iloc[1]["destination_ip"], "UNKNOWN")

    def _structured_logs(self):
        valid_row = {
            "date": "2008-11-09",
            "time": "20:35:19",
            "pid": "145",
            "log_level": "INFO",
            "component": "dfs.DataNode$PacketResponder",
            "event_type": "RECEIVED_BLOCK",
            "block_id": "blk_-1608999687919862906",
            "source_ip": "10.250.10.6",
            "destination_ip": None,
            "block_size": "91178",
            "raw_message": "Received block blk_-1608999687919862906 of size 91178",
        }
        missing_optional_row = {
            "date": "2008-11-09",
            "time": "21:01:00",
            "pid": "19",
            "log_level": "ERROR",
            "component": "dfs.FSNamesystem",
            "event_type": "REPLICATE_BLOCK",
            "block_id": "blk_7503483334202473044",
            "source_ip": None,
            "destination_ip": None,
            "block_size": None,
            "raw_message": "BLOCK* ask 10.250.14.224:50010 to replicate",
        }
        invalid_required_row = {
            "date": "not-a-date",
            "time": "21:01:00",
            "pid": "bad-pid",
            "log_level": "INFO",
            "component": "dfs.FSNamesystem",
            "event_type": "OTHER",
            "block_id": "",
            "source_ip": None,
            "destination_ip": None,
            "block_size": None,
            "raw_message": "bad row",
        }
        return pd.DataFrame(
            [valid_row, valid_row.copy(), missing_optional_row, invalid_required_row]
        )

    def _temp_output_path(self, filename):
        return Path(tempfile.gettempdir()) / filename


class HDFSFeatureExtractorTest(unittest.TestCase):
    def test_extractor_validates_clean_schema(self):
        frame = self._cleaned_logs().drop(columns=["block_id"])

        with self.assertRaises(ValueError):
            HDFSFeatureExtractor().extract(frame, self._temp_output_path("features.csv"))

    def test_extractor_generates_numeric_features_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "features.csv"
            features = HDFSFeatureExtractor().extract(self._cleaned_logs(), output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(list(features.columns), FEATURE_COLUMNS)
            self.assertTrue(
                all(pd.api.types.is_numeric_dtype(features[column]) for column in features)
            )

            self.assertEqual(len(features), 2)
            self.assertEqual(features.loc[0, "hour_of_day"], 20)
            self.assertEqual(features.loc[0, "is_error"], 1)
            self.assertEqual(features.loc[0, "event_frequency"], 2)
            self.assertEqual(features.loc[1, "hour_of_day"], 21)
            self.assertEqual(features.loc[1, "is_error"], 0)
            self.assertEqual(features.loc[1, "event_frequency"], 1)

    def test_extractor_writes_one_row_per_block_id_with_no_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "features.csv"
            HDFSFeatureExtractor().extract(self._cleaned_logs(), output_path)

            on_disk = pd.read_csv(output_path)

            self.assertEqual(list(on_disk.columns)[0], "block_id")
            self.assertEqual(list(on_disk.columns)[1:], FEATURE_COLUMNS)
            self.assertFalse(on_disk["block_id"].duplicated().any())
            self.assertEqual(
                sorted(on_disk["block_id"]),
                sorted(self._cleaned_logs()["block_id"].unique()),
            )

    def _cleaned_logs(self):
        return pd.DataFrame(
            [
                {
                    "datetime": "2008-11-09 20:35:19",
                    "log_level": "INFO",
                    "component": "dfs.DataNode$PacketResponder",
                    "event_type": "RECEIVED_BLOCK",
                    "block_id": "blk_-1608999687919862906",
                    "source_ip": "10.250.10.6",
                    "destination_ip": "UNKNOWN",
                    "block_size": 91178,
                },
                {
                    "datetime": "2008-11-09 20:35:20",
                    "log_level": "ERROR",
                    "component": "dfs.DataNode$PacketResponder",
                    "event_type": "RECEIVED_BLOCK",
                    "block_id": "blk_-1608999687919862906",
                    "source_ip": "10.250.19.102",
                    "destination_ip": "UNKNOWN",
                    "block_size": 91178,
                },
                {
                    "datetime": "2008-11-09 21:01:00",
                    "log_level": "INFO",
                    "component": "dfs.FSNamesystem",
                    "event_type": "REPLICATE_BLOCK",
                    "block_id": "blk_7503483334202473044",
                    "source_ip": "UNKNOWN",
                    "destination_ip": "10.251.215.16",
                    "block_size": 0,
                },
            ]
        )

    def _temp_output_path(self, filename):
        return Path(tempfile.gettempdir()) / filename


if __name__ == "__main__":
    unittest.main()
