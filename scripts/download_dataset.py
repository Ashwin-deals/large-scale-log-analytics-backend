"""Download the raw datasets from Hugging Face into data/raw/<source>/.

None of the raw logs are committed to this repo — they're hosted on Hugging Face instead:
https://huggingface.co/datasets/Ashwin-deals/large-scale-log-analytics-hdfs-data

- hdfs: HDFS.log (~1.5GB) and anomaly_label.csv, at the root of the dataset repo.
- openstack, apache, django: one folder each in the dataset repo, mirroring
  data/raw/<source>/ (apache/injected/ and the django files are the outputs of
  scripts/inject_apache_anomalies.py and scripts/generate_django_logs.py, seed 42).
- cloudtrail: NOT re-hosted. We found no licence for the flAWS.cloud CloudTrail
  logs, so they come from the author's original archive (Scott Piper / Summit
  Route, ~250MB) and each of the 20 files is checked against a known SHA-256.

If the dataset is private or gated, set HF_TOKEN in .env (or the environment).

--with-derived also fetches what a fresh clone needs to serve the dashboard
without rebuilding anything (the derived/ folder of the dataset repo):

- derived/hdfs/features.csv, block_metadata.csv, dataset_stats.json
  -> data/features/features.csv, data/processed/block_metadata.csv, data/processed/dataset_stats.json
- derived/vocab/<source>/category_encoders.json
  -> data/features/category_encoders.json (hdfs) or data/features/<source>/category_encoders.json.
  Every model is stamped with the vocabulary it was trained on and scoring
  refuses to run without it, so these must travel with the data.
  CloudTrail's is derived from the CloudTrail logs, so it is not hosted either:
  `python3 scripts/run_source_pipeline.py --source cloudtrail` rebuilds it
  (deterministically, so the committed models still match).

Usage:
    python3 scripts/download_dataset.py                    # HDFS only, as before
    python3 scripts/download_dataset.py --source openstack
    python3 scripts/download_dataset.py --source all --with-derived
"""

import argparse
import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

# Without this the HF_TOKEN in .env is invisible to the script, and a gated
# dataset fails with a 401 even though the token is configured.
load_dotenv()

REPO_ID = "Ashwin-deals/large-scale-log-analytics-hdfs-data"
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
DEST_DIR = os.path.join(RAW_DIR, "hdfs")
FILES = ["HDFS.log", "anomaly_label.csv"]
FOLDER_SOURCES = ["openstack", "apache", "django"]  # hosted on Hugging Face
ALL_SOURCES = ["openstack", "apache", "cloudtrail", "django"]

# The flAWS.cloud CloudTrail logs, from the author's own download (overridable for tests).
FLAWS_URL = os.getenv("FLAWS_URL", "https://summitroute.com/downloads/flaws_cloudtrail_logs.tar")
FLAWS_SHA256 = {
    "flaws_cloudtrail00.json.gz": "62dadeb453cefd0ac3a863003fa340b09375380daac3009855907234fa42845e",
    "flaws_cloudtrail01.json.gz": "0a41ae88793e75d027031ce5e9a2f613a458cdeb984a2e5ec95c4f08b7b661a4",
    "flaws_cloudtrail02.json.gz": "9276398fe6a7434b5505b0ce50be1e1d16b066fa304659f1b234af0c315e6ad5",
    "flaws_cloudtrail03.json.gz": "b3f11ca8a062f23048879f07717e4c35f225562dab051f20483cfe1100ae5c62",
    "flaws_cloudtrail04.json.gz": "bab0f15bfe4a77c8557377704e745a5e4f00530c96afc5e20963c8dbf3055df4",
    "flaws_cloudtrail05.json.gz": "832b45f26ddcef4e1490452152b067f653016c91020945bedbe13172f1626094",
    "flaws_cloudtrail06.json.gz": "b6515ac7c49bf5f91d312d9fecb2727196b32c3617a2ef3a9c545143341ab1ae",
    "flaws_cloudtrail07.json.gz": "5e360b277e6b0ddbcabb378b49b2c34d88f8c26ab9f029774d479078f3d8232b",
    "flaws_cloudtrail08.json.gz": "440dd073958db7df96525f00d0e74c7156b770e0c743c659d792c069937b9331",
    "flaws_cloudtrail09.json.gz": "eb33a66b5c6bc4da3cb381f50dfc75f300cea84c54388872818b5d768f759570",
    "flaws_cloudtrail10.json.gz": "c424667317bb38899fb6099822519ff3f8c7170ce04c22318ceee45067572222",
    "flaws_cloudtrail11.json.gz": "2302322e6fb4e8a811aaa6d22ecbe7f66866514f01f81fbfd8498aba9ea63266",
    "flaws_cloudtrail12.json.gz": "2beb7dd3065885d466287b560016d53242bab999881d49e26fe402e63e734465",
    "flaws_cloudtrail13.json.gz": "bea798bc93993fd0eb284445488dadd97b93ba5885817087ce711eecfb8a99f6",
    "flaws_cloudtrail14.json.gz": "6cfd9f02bc50120ae0cd9d685a80088e0343e4ac643724604f7161989d2750cd",
    "flaws_cloudtrail15.json.gz": "54b1bb0c990c46d445b7674e1e5343ef730f0848d5012a6c3894e58311daab87",
    "flaws_cloudtrail16.json.gz": "0b575d07597988c39da91e5c55ec4b5493746012bc64e562320cf3748e363d35",
    "flaws_cloudtrail17.json.gz": "7530a1865425c8476e124dbeb04fd86172165362c931d835d9323c6c173af464",
    "flaws_cloudtrail18.json.gz": "2cb4d779ecd05c6c24b595103dcd184b915162bc00f6662269b163d66978500a",
    "flaws_cloudtrail19.json.gz": "1bcf77a96388ba206ae0e960403b87e8e0000299dfcfc5e0e2d35906a06f24b2",
}


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


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_cloudtrail(dest=None, url=None, expected=None):
    """Fetch the flAWS archive, keep only the 20 expected .json.gz files
    (written by base name, so nothing in the archive can choose its own path)
    and verify each against its known SHA-256."""
    dest = dest or os.path.join(RAW_DIR, "cloudtrail")
    url = url or FLAWS_URL
    expected = expected or FLAWS_SHA256
    os.makedirs(dest, exist_ok=True)
    present = [n for n in expected if os.path.exists(os.path.join(dest, n)) and _sha256(os.path.join(dest, n)) == expected[n]]
    if len(present) == len(expected):
        print(f"cloudtrail: all {len(expected)} files already present and verified.")
        return
    print(f"Downloading CloudTrail logs from {url} ...")
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "flaws.tar")
        request = urllib.request.Request(url, headers={"User-Agent": "morphguard-download/1.0"})
        with urllib.request.urlopen(request) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if member.isfile() and name in expected:
                    with tar.extractfile(member) as src, open(os.path.join(tmp, name), "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 20)
        bad = [n for n in expected if not os.path.exists(os.path.join(tmp, n)) or _sha256(os.path.join(tmp, n)) != expected[n]]
        if bad:
            raise SystemExit(f"cloudtrail: {len(bad)} file(s) missing or with an unexpected checksum: {', '.join(sorted(bad))}")
        for name in expected:
            shutil.move(os.path.join(tmp, name), os.path.join(dest, name))
    print(f"  -> {dest} ({len(expected)} files verified)")


DERIVED_HDFS = {
    "derived/hdfs/features.csv": "data/features/features.csv",
    "derived/hdfs/block_metadata.csv": "data/processed/block_metadata.csv",
    "derived/hdfs/dataset_stats.json": "data/processed/dataset_stats.json",
}


def vocab_target(source):
    return "data/features/category_encoders.json" if source == "hdfs" else f"data/features/{source}/category_encoders.json"


def derived_files(source):
    """(path in the dataset repo, path in this checkout) for one source."""
    files = list(DERIVED_HDFS.items()) if source == "hdfs" else []
    files.append((f"derived/vocab/{source}/category_encoders.json", vocab_target(source)))
    return files


def download_derived(source, token):
    if source == "cloudtrail":
        print("cloudtrail vocabulary is not hosted; run scripts/run_source_pipeline.py --source cloudtrail to rebuild it.")
        return
    root = os.path.dirname(os.path.dirname(RAW_DIR))
    with tempfile.TemporaryDirectory() as tmp:
        for remote, local in derived_files(source):
            print(f"Downloading {remote} ...")
            fetched = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=remote, local_dir=tmp, token=token)
            target = os.path.join(root, local)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(fetched, target)
            print(f"  -> {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["hdfs", *ALL_SOURCES, "all"], default="hdfs")
    parser.add_argument("--with-derived", action="store_true",
                        help="also fetch derived files (HDFS features/metadata, every source's vocabulary)")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN") or None
    if token:
        print("Using HF_TOKEN from the environment.")

    if args.source in ("hdfs", "all"):
        download_hdfs(token)
    for source in FOLDER_SOURCES:
        if args.source in (source, "all"):
            download_folder(source, token)
    if args.source in ("cloudtrail", "all"):
        download_cloudtrail()
    if args.with_derived:
        for source in ["hdfs", *ALL_SOURCES]:
            if args.source in (source, "all"):
                download_derived(source, token)


if __name__ == "__main__":
    main()
