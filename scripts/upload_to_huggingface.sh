#!/usr/bin/env bash
# Upload every dataset, the HDFS derived files and each source's vocabulary to
# the Hugging Face dataset repo, in the layout scripts/download_dataset.py
# --with-derived expects. Run from the backend folder after `hf auth login`
# with a WRITE token. HDFS.log and anomaly_label.csv are already there.
# Check the licences in docs/huggingface/README.md before running.
set -euo pipefail
REPO=${REPO:-Ashwin-deals/large-scale-log-analytics-hdfs-data}
# A failed upload stops the script with its name (set -e); only a source
# that is genuinely absent on this machine is skipped.
up() { hf upload "$REPO" "$1" "$2" --repo-type=dataset || { echo "FAILED: $1 -> $2" >&2; exit 1; }; }

# derived/hdfs must be the full corpus: a sampled features.csv (e.g. the
# 150,000-block representative sample) would silently replace it for everyone.
blocks=$(($(wc -l < data/features/features.csv) - 1))
if [ "$blocks" -ne 575061 ]; then
  echo "data/features/features.csv has $blocks blocks, not the full 575,061. Run this on the machine with the full HDFS features." >&2
  exit 1
fi

# SKIP_SOURCES="cloudtrail apache" leaves out sources whose licence does not allow re-hosting.
skipped() { case " ${SKIP_SOURCES:-} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

for s in openstack apache cloudtrail django; do
  if skipped "$s"; then
    echo "skip $s: in SKIP_SOURCES"
  elif [ -d "data/raw/$s" ]; then
    up "data/raw/$s" "$s"
  else
    echo "skip $s: data/raw/$s missing"
  fi
done
up data/features/features.csv derived/hdfs/features.csv
up data/processed/block_metadata.csv derived/hdfs/block_metadata.csv
up data/processed/dataset_stats.json derived/hdfs/dataset_stats.json
up data/features/category_encoders.json derived/vocab/hdfs/category_encoders.json
for s in openstack apache cloudtrail django; do
  f="data/features/$s/category_encoders.json"
  if [ -f "$f" ]; then
    up "$f" "derived/vocab/$s/category_encoders.json"
  else
    echo "skip vocab $s: $f missing"
  fi
done
echo "Upload complete."
up docs/huggingface/README.md README.md
