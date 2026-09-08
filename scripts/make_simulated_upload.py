"""Build a simulated "new logs arriving from the cluster" file for testing ingestion.

Why this is needed: every block in HDFS.log is already part of the deployed
575,061-block dataset. A block_id identifies one real HDFS block, so uploading
any real slice of that log correctly adds *nothing* to the dashboard totals —
counting the same block twice would make the cluster look bigger than it is.
Exercising the "an upload extends the dashboard" path therefore needs blocks
the dataset has genuinely never seen.

This takes real HDFS log lines and rewrites ONLY their block identifiers to
fresh, non-colliding ones, checked against anomaly_label.csv so a generated id
can never shadow a real block. Everything else — components, IPs, timestamps,
event mix, line structure — is exactly what the cluster produced.

So: synthetic *identity*, real log content. It is for testing the ingestion
path only. These blocks have no ground-truth label, so never quote model
accuracy from them.

Usage:
    python3 scripts/make_simulated_upload.py
    python3 scripts/make_simulated_upload.py --blocks 800 --out data/testing/simulated_new_logs.log
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from parser.hdfs_parser import BLOCK_ID_PATTERN

DEFAULT_SOURCE = Path("data/testing/review_dataset_raw.log")
DEFAULT_OUT = Path("data/testing/simulated_new_logs.log")
LABELS_PATH = Path("data/raw/hdfs/anomaly_label.csv")

# Well clear of the real id space, which keeps generated ids readable as
# "obviously not from the 2008 trace" when someone eyeballs the dashboard.
ID_BASE = 7_700_000_000_000_000_000


def build(source: Path, out: Path, block_count: int) -> None:
    known = set(pd.read_csv(LABELS_PATH)["BlockId"])

    # Pick the first N distinct blocks in order of appearance, so each one
    # keeps its full set of lines rather than a truncated lifecycle.
    selected: dict[str, str] = {}
    next_id = ID_BASE
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = BLOCK_ID_PATTERN.search(line)
            if not match:
                continue
            block_id = match.group(0)
            if block_id in selected or len(selected) >= block_count:
                continue
            candidate = f"blk_{next_id}"
            while candidate in known:
                next_id += 1
                candidate = f"blk_{next_id}"
            selected[block_id] = candidate
            next_id += 1

    if not selected:
        raise SystemExit(f"No block ids found in {source}")

    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace") as src, out.open("w", encoding="utf-8") as dst:
        for line in src:
            match = BLOCK_ID_PATTERN.search(line)
            if not match:
                continue
            original = match.group(0)
            replacement = selected.get(original)
            if replacement is None:
                continue
            # Replace every mention: a single line can name the block more
            # than once, and a half-renamed line would split one block in two.
            rewritten = line.replace(original, replacement)
            dst.write(rewritten if rewritten.endswith("\n") else rewritten + "\n")
            written += 1

    print(f"Source:            {source}")
    print(f"Blocks simulated:  {len(selected):,} (new ids, none colliding with the {len(known):,} real blocks)")
    print(f"Lines written:     {written:,}")
    print(f"Wrote:             {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--blocks", type=int, default=500)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"{args.source} not found.")
    build(args.source, args.out, args.blocks)


if __name__ == "__main__":
    main()
