"""Log sources beyond HDFS: OpenStack, Apache, CloudTrail, Django.

Each source's parser (parser/<source>_parser.py) emits COMMON_LOG_SCHEMA, so
the same HDFSDataCleaner -> HDFSFeatureExtractor -> Isolation Forest -> GA
-> Smart Storage -> early-warning chain runs on every source. Each source has
its own category vocabulary, models, version lineage and metrics:

    data/processed/<source>/     structured.csv, cleaned.csv
    data/features/<source>/      features.csv, labels.csv, category_encoders.json
    data/models/<source>/        isolation_forest_v{N}.pkl, current_version.json, early_warning/
    data/evaluation/<source>_metrics/, data/evaluation/<source>_report.json
    data/storage/<source>/       policy_best.json, storage_report.json

The HDFS files (data/models/isolation_forest_v2.pkl, data/features/
category_encoders.json, ...) are never read or written by this package.

  sources.py   the registry: inputs, parser, labels, caveats per source
  labels.py    label loaders, including CloudTrail's documented weak rule
  pipeline.py  ingest -> train/promote -> smart storage -> early warning -> report
  upload.py    score an uploaded file of any source against that source's model
"""
