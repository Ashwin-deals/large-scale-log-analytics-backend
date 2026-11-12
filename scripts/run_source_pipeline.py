"""Run the full pipeline for one or more non-HDFS log sources.

parse -> clean -> extract -> V1 baseline -> GA candidate -> promote -> Smart
Storage -> early warning -> data/evaluation/<source>_report.json. Each source
keeps its own vocabulary, models, versions and metrics; HDFS is never touched.
See multi_source/pipeline.py.

Prerequisites:
    apache  python3 scripts/inject_apache_anomalies.py   (writes the INJECTED copy + labels)
    django  python3 scripts/generate_django_logs.py      (writes the SYNTHETIC log + labels)

Usage:
    python3 scripts/run_source_pipeline.py --source openstack
    python3 scripts/run_source_pipeline.py --source all
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from multi_source.pipeline import run  # noqa: E402
from multi_source.sources import SOURCES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True, choices=list(SOURCES) + ["all"])
    args = parser.parse_args()
    for source in (list(SOURCES) if args.source == "all" else [args.source]):
        run(source)


if __name__ == "__main__":
    main()
