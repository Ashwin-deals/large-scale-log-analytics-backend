#!/usr/bin/env bash
# Upload every dataset, the HDFS derived files and each source's vocabulary to
# the Hugging Face dataset repo, in the layout scripts/download_dataset.py
# --with-derived expects. Run from the backend folder after `hf auth login`
# with a WRITE token. HDFS.log and anomaly_label.csv are already there.
# Check the licences in docs/huggingface/README.md before running.
set -euo pipefail
REPO=${REPO:-Ashwin-deals/large-scale-log-analytics-hdfs-data}
up() { hf upload "$REPO" "$1" "$2" --repo-type=dataset; }

# derived/hdfs must be the full corpus: a sampled features.csv (e.g. the
# 150,000-block representative sample) would silently replace it for everyone.
blocks=$(($(wc -l < data/features/features.csv) - 1))
if [ "$blocks" -ne 575061 ]; then
  echo "data/features/features.csv has $blocks blocks, not the full 575,061. Run this on the machine with the full HDFS features." >&2
  exit 1
fi

for s in openstack apache cloudtrail django; do
  [ -d "data/raw/$s" ] && up "data/raw/$s" "$s" || echo "skip $s: data/raw/$s missing"
done
up data/features/features.csv derived/hdfs/features.csv
up data/processed/block_metadata.csv derived/hdfs/block_metadata.csv
up data/processed/dataset_stats.json derived/hdfs/dataset_stats.json
up data/features/category_encoders.json derived/vocab/hdfs/category_encoders.json
for s in openstack apache cloudtrail django; do
  f="data/features/$s/category_encoders.json"
  [ -f "$f" ] && up "$f" "derived/vocab/$s/category_encoders.json" || echo "skip vocab $s: $f missing"
done
up docs/huggingface/README.md README.md
