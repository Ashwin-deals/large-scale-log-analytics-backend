"""Download the raw HDFS dataset from Hugging Face into data/raw/hdfs/.

The raw log (HDFS.log, ~1.5GB) and its anomaly labels (anomaly_label.csv)
aren't committed to this repo — they're hosted on Hugging Face instead:
https://huggingface.co/datasets/Ashwin-deals/large-scale-log-analytics-hdfs-data

If the dataset is private or gated, set HF_TOKEN in .env (or the environment).

Usage:
    python3 scripts/download_dataset.py
"""

import os

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# Without this the HF_TOKEN in .env is invisible to the script, and a gated
# dataset fails with a 401 even though the token is configured.
load_dotenv()

REPO_ID = "Ashwin-deals/large-scale-log-analytics-hdfs-data"
DEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "hdfs")
FILES = ["HDFS.log", "anomaly_label.csv"]


def main():
    token = os.getenv("HF_TOKEN") or None
    if token:
        print("Using HF_TOKEN from the environment.")

    os.makedirs(DEST_DIR, exist_ok=True)
    for filename in FILES:
        print(f"Downloading {filename} ...")
        path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename, local_dir=DEST_DIR, token=token
        )
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
