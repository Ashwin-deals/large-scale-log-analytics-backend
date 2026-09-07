import tempfile
import unittest
from pathlib import Path

import pandas as pd

from parser.base_parser import COMMON_LOG_SCHEMA
from parser.hdfs_parser import HDFSParser


class HDFSParserTest(unittest.TestCase):
    def test_hdfs_parser_returns_common_schema(self):
        lines = [
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010",
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6",
            "081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_-1608999687919862906 terminating",
            "081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/job.jar. blk_-1608999687919862906",
            "081109 203519 29 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock: blockMap updated: 10.250.10.6:50010 is added to blk_-1608999687919862906 size 91178",
            "081109 203523 148 INFO dfs.DataNode$DataXceiver: 10.250.11.100:50010 Served block blk_-3544583377289625738 to /10.250.19.102",
            "081109 203521 19 INFO dfs.DataNode: 10.250.14.224:50010 Starting thread to transfer block blk_-1608999687919862906 to 10.251.215.16:50010, 10.251.71.193:50010",
            "081109 203521 147 INFO dfs.DataNode$DataTransfer: 10.250.14.224:50010:Transmitted block blk_-1608999687919862906 to /10.251.215.16:50010",
            "081109 203521 19 INFO dfs.FSNamesystem: BLOCK* ask 10.250.14.224:50010 to replicate blk_-1608999687919862906 to datanode(s) 10.251.215.16:50010",
            "081109 203521 99 WARN dfs.Unknown: Something unexpected happened",
        ]

        frame = self._parse_lines(lines)

        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(list(frame.columns), COMMON_LOG_SCHEMA)
        self.assertEqual(
            frame["event_type"].tolist(),
            [
                "RECEIVE_BLOCK",
                "RECEIVED_BLOCK",
                "PACKET_TERMINATED",
                "ALLOCATE_BLOCK",
                "STORE_BLOCK",
                "SERVE_BLOCK",
                "START_TRANSFER",
                "TRANSMIT_BLOCK",
                "REPLICATE_BLOCK",
                "OTHER",
            ],
        )

    def test_hdfs_parser_extracts_structured_fields(self):
        frame = self._parse_lines(
            [
                "081109 203519 145 INFO dfs.DataNode$PacketResponder: "
                "Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6"
            ]
        )

        row = frame.iloc[0]

        self.assertEqual(row["date"], "2008-11-09")
        self.assertEqual(row["time"], "20:35:19")
        self.assertEqual(row["pid"], 145)
        self.assertEqual(row["log_level"], "INFO")
        self.assertEqual(row["component"], "dfs.DataNode$PacketResponder")
        self.assertEqual(row["block_id"], "blk_-1608999687919862906")
        self.assertEqual(row["source_ip"], "10.250.10.6")
        self.assertIsNone(row["destination_ip"])
        self.assertEqual(row["block_size"], 91178)
        self.assertEqual(
            row["raw_message"],
            "Received block blk_-1608999687919862906 of size 91178 from /10.250.10.6",
        )

    def test_hdfs_parser_extracts_transfer_ips(self):
        frame = self._parse_lines(
            [
                "081109 203521 147 INFO dfs.DataNode$DataTransfer: "
                "10.250.14.224:50010:Transmitted block blk_-1608999687919862906 "
                "to /10.251.215.16:50010",
                "081109 203521 19 INFO dfs.DataNode: "
                "10.250.14.224:50010 Starting thread to transfer block "
                "blk_-1608999687919862906 to 10.251.215.16:50010, "
                "10.251.71.193:50010",
            ]
        )

        self.assertEqual(frame.iloc[0]["source_ip"], "10.250.14.224")
        self.assertEqual(frame.iloc[0]["destination_ip"], "10.251.215.16")
        self.assertEqual(frame.iloc[1]["source_ip"], "10.250.14.224")
        self.assertEqual(frame.iloc[1]["destination_ip"], "10.251.215.16")

    def _parse_lines(self, lines):
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "HDFS.log"
            log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return HDFSParser().parse(str(log_file))


if __name__ == "__main__":
    unittest.main()
