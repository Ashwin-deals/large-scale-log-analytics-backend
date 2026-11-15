"""Measure the longest quiet gap inside each HDFS block, in log time.

The stream processor closes a block once it has been quiet for the source's
idle timeout (streaming/sources.py IDLE_TIMEOUTS). If the timeout is shorter
than a block's longest gap between two of its own lines, the block is closed
early and every later line counts as late. This script measures that gap for
every block in the full HDFS.log so the HDFS timeout can be chosen from data.

Blocks are keyed the way the producer keys them: by the first blk_ id in the
line (streaming/producer.py block_key). Timestamps are the line's leading
"yymmdd HHMMSS". A block with a single line has a gap of 0.

Writes data/evaluation/hdfs_block_gap_stats.json: percentiles of the per-block
longest gap, and the share of blocks each candidate timeout would cover.

Usage:
    python3 scripts/measure_block_gaps.py
    python3 scripts/measure_block_gaps.py --raw-log data/raw/hdfs/HDFS.log
"""

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_LOG = ROOT_DIR / "data/raw/hdfs/HDFS.log"
OUTPUT = ROOT_DIR / "data/evaluation/hdfs_block_gap_stats.json"
BLOCK_ID = re.compile(r"\bblk_-?\d+\b")
PERCENTILES = (50, 90, 95, 99, 99.9)
CANDIDATE_TIMEOUTS = (3600, 2 * 3600, 6 * 3600, 12 * 3600, 24 * 3600)


def longest_gaps(raw_log: Path) -> tuple[dict, dict]:
    """block id -> longest gap (seconds) between consecutive lines, in file order."""
    day_start: dict[str, int] = {}
    last_seen: dict[str, int] = {}
    gap: dict[str, int] = {}
    stats = {"lines": 0, "lines_with_block": 0, "time_inversions": 0}
    with raw_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stats["lines"] += 1
            match = BLOCK_ID.search(line)
            if match is None:
                continue
            day, clock = line[:6], line[7:13]
            if not (day.isdigit() and clock.isdigit()):
                continue
            if day not in day_start:
                day_start[day] = int(datetime.strptime(day, "%y%m%d").timestamp())
            t = day_start[day] + int(clock[:2]) * 3600 + int(clock[2:4]) * 60 + int(clock[4:6])
            stats["lines_with_block"] += 1
            block = match.group(0)
            previous = last_seen.get(block)
            if previous is None:
                gap[block] = 0
            else:
                step = t - previous
                if step < 0:
                    stats["time_inversions"] += 1
                elif step > gap[block]:
                    gap[block] = step
            last_seen[block] = t
    return gap, stats


def coverage_for(gaps: np.ndarray, timeout: int) -> float:
    """Share of blocks the processor would never close early: every gap inside
    the block is at most the timeout (the processor closes on a gap > timeout)."""
    return float((gaps <= timeout).mean())


def smallest_timeout_covering(gaps: np.ndarray, share: float) -> int:
    """Smallest whole-second timeout that leaves at least `share` of blocks unbroken."""
    ordered = np.sort(gaps)
    index = max(0, math.ceil(share * len(ordered)) - 1)
    return int(ordered[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-log", type=Path, default=RAW_LOG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--target", type=float, default=0.99, help="share of blocks the chosen timeout must cover")
    args = parser.parse_args()

    if not args.raw_log.exists():
        sys.exit(f"{args.raw_log} not found: run python3 scripts/download_dataset.py first")

    print(f"Scanning {args.raw_log} ...", flush=True)
    gap, stats = longest_gaps(args.raw_log)
    gaps = np.fromiter(gap.values(), dtype=np.int64, count=len(gap))
    chosen = smallest_timeout_covering(gaps, args.target)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "raw_log": str(args.raw_log.relative_to(ROOT_DIR) if args.raw_log.is_relative_to(ROOT_DIR) else args.raw_log),
        "definition": "per block, the longest log-time gap in seconds between two consecutive lines of that block "
                      "(file order, keyed by the first blk_ id in the line, as the producer keys messages)",
        "scan": stats,
        "blocks": int(len(gaps)),
        "single_line_blocks": int((gaps == 0).sum()),
        "longest_gap_seconds": {
            **{f"p{p:g}": float(np.percentile(gaps, p)) for p in PERCENTILES},
            "max": int(gaps.max()),
            "mean": float(gaps.mean()),
        },
        "coverage_by_timeout": {str(t): coverage_for(gaps, t) for t in CANDIDATE_TIMEOUTS},
        "target_share": args.target,
        "smallest_timeout_covering_target_seconds": chosen,
        "coverage_at_chosen": coverage_for(gaps, chosen),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{stats['lines']:,} lines, {stats['lines_with_block']:,} with a block id, "
          f"{len(gaps):,} blocks, {stats['time_inversions']:,} time inversions")
    for key, value in report["longest_gap_seconds"].items():
        print(f"  {key:>6}: {value:>10,.0f} s  ({value / 3600:.2f} h)")
    for timeout, share in report["coverage_by_timeout"].items():
        print(f"  timeout {int(timeout):>6,} s covers {share:.4%} of blocks")
    print(f"Smallest timeout covering {args.target:.0%}: {chosen:,} s ({chosen / 3600:.2f} h), "
          f"covers {report['coverage_at_chosen']:.4%}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
