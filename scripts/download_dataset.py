"""Download the raw datasets from Hugging Face into data/raw/<source>/.

None of the raw logs are committed to this repo — they're hosted on Hugging Face instead:
https://huggingface.co/datasets/Ashwin-deals/large-scale-log-analytics-hdfs-data

- hdfs: HDFS.log (~1.5GB) and anomaly_label.csv, at the root of the dataset repo.
- openstack, apache, cloudtrail, django: one folder each in the dataset repo, mirroring
  data/raw/<source>/ (apache/injected/ and the django files are the outputs of
  scripts/inject_apache_anomalies.py and scripts/generate_django_logs.py, seed 42).

If the dataset is private or gated, set HF_TOKEN in .env (or the environment).

Usage:
    python3 scripts/download_dataset.py                    # HDFS only, as before
    python3 scripts/download_dataset.py --source openstack
    python3 scripts/download_dataset.py --source all
"""

import argparse
import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

# Without this the HF_TOKEN in .env is invisible to the script, and a gated
# dataset fails with a 401 even though the token is configured.
load_dotenv()

REPO_ID = "Ashwin-deals/large-scale-log-analytics-hdfs-data"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
DEST_DIR = os.path.join(RAW_DIR, "hdfs")
FILES = ["HDFS.log", "anomaly_label.csv"]
FOLDER_SOURCES = ["openstack", "apache", "cloudtrail", "django"]


def download_hdfs(token):
    os.makedirs(DEST_DIR, exist_ok=True)
    for filename in FILES:
        print(f"Downloading {filename} ...")
        path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=DEST_DIR, token=token
        )
        print(f"  -> {path}")


def download_folder(source, token):
    print(f"Downloading {source}/ ...")
    snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", allow_patterns=f"{source}/**", local_dir=RAW_DIR, token=token
    )
    print(f"  -> {os.path.join(RAW_DIR, source)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["hdfs", *FOLDER_SOURCES, "all"], default="hdfs")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or None
    if token:
        print("Using HF_TOKEN from the environment.")

    if args.source in ("hdfs", "all"):
        download_hdfs(token)
    for source in FOLDER_SOURCES:
        if args.source in (source, "all"):
            download_folder(source, token)


if __name__ == "__main__":
    main()
