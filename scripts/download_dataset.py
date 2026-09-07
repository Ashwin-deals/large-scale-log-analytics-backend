"""Download the raw HDFS dataset from Hugging Face into data/raw/hdfs/.

The raw log (HDFS.log, ~1.5GB) and its anomaly labels (anomaly_label.csv)
aren't committed to this repo — they're hosted on Hugging Face instead:
https://huggingface.co/datasets/Ashwin-deals/large-scale-log-analytics-hdfs-data

Usage:
    python3 scripts/download_dataset.py
"""

import os

from huggingface_hub import hf_hub_download

REPO_ID = "Ashwin-deals/large-scale-log-analytics-hdfs-data"
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "hdfs")
FILES = ["HDFS.log", "anomaly_label.csv"]


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    for filename in FILES:
        print(f"Downloading {filename} ...")
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=DEST_DIR)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
