"""
Verification-only check that data/features/features.csv can be safely joined
with data/raw/hdfs/anomaly_label.csv before any model training happens.

No model is trained here. If verification passes, the merged/labeled frame is
written to data/features/features_labeled.csv; otherwise nothing is written.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DROP_FLAG_THRESHOLD_PCT = 1.0
BLOCK_ID_NAME_CANDIDATES = ["block_id", "blockid", "blk_id"]
LABEL_NAME_CANDIDATES = ["label", "anomaly", "class"]


def find_column(df: pd.DataFrame, name_candidates: list[str], value_pattern: "re.Pattern | None" = None) -> str | None:
    """
    Finds a column by exact/lowercased name match first, falling back to a
    column whose sample values match `value_pattern`.
    """
    lower_map = {column.lower(): column for column in df.columns}
    for candidate in name_candidates:
        if candidate in lower_map:
            return lower_map[candidate]

    for column in df.columns:
        if any(candidate in column.lower() for candidate in name_candidates):
            return column

    if value_pattern is not None:
        for column in df.columns:
            sample = df[column].dropna().astype(str).head(20)
            if len(sample) > 0 and sample.str.match(value_pattern).all():
                return column

    return None


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def verify(features_path: Path, labels_path: Path, output_path: Path) -> bool:
    reasons: list[str] = []

    section("Loading files")
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path, dtype=str)
    print(f"features.csv:      {features_df.shape[0]:,} rows x {features_df.shape[1]} columns")
    print(f"  columns: {list(features_df.columns)}")
    print(f"anomaly_label.csv: {labels_df.shape[0]:,} rows x {labels_df.shape[1]} columns")
    print(f"  columns: {list(labels_df.columns)}")

    section("Identifying block ID columns")
    block_id_pattern = re.compile(r"^blk_-?\d+$")
    features_id_col = find_column(features_df, BLOCK_ID_NAME_CANDIDATES, block_id_pattern)
    labels_id_col = find_column(labels_df, BLOCK_ID_NAME_CANDIDATES, block_id_pattern)

    if features_id_col:
        print(f"features.csv id column:      '{features_id_col}'")
        print(f"  sample IDs: {features_df[features_id_col].dropna().astype(str).head(5).tolist()}")
    else:
        print("features.csv id column:      NOT FOUND")
        reasons.append(
            "features.csv has no block ID column to join on "
            f"(columns present: {list(features_df.columns)})"
        )

    if labels_id_col:
        print(f"anomaly_label.csv id column: '{labels_id_col}'")
        print(f"  sample IDs: {labels_df[labels_id_col].dropna().astype(str).head(5).tolist()}")
    else:
        print("anomaly_label.csv id column: NOT FOUND")
        reasons.append(
            "anomaly_label.csv has no block ID column to join on "
            f"(columns present: {list(labels_df.columns)})"
        )

    section("Checking for duplicate IDs")
    if features_id_col:
        dup_count = features_df[features_id_col].duplicated().sum()
        if dup_count > 0:
            print(f"FLAG: features.csv has {dup_count:,} duplicate '{features_id_col}' values")
            reasons.append(f"features.csv has {dup_count:,} duplicate IDs")
        else:
            print(f"features.csv: no duplicate '{features_id_col}' values")
    else:
        print("features.csv: skipped (no id column)")

    if labels_id_col:
        dup_count = labels_df[labels_id_col].duplicated().sum()
        if dup_count > 0:
            print(f"FLAG: anomaly_label.csv has {dup_count:,} duplicate '{labels_id_col}' values")
            reasons.append(f"anomaly_label.csv has {dup_count:,} duplicate IDs")
        else:
            print(f"anomaly_label.csv: no duplicate '{labels_id_col}' values")
    else:
        print("anomaly_label.csv: skipped (no id column)")

    merged_df: pd.DataFrame | None = None
    label_col: str | None = None

    section("Inner join on block ID")
    if features_id_col and labels_id_col:
        rows_before = len(features_df)
        merged_df = features_df.merge(
            labels_df, left_on=features_id_col, right_on=labels_id_col, how="inner"
        )
        rows_after = len(merged_df)
        dropped = rows_before - rows_after
        dropped_pct = (dropped / rows_before * 100) if rows_before else 0.0
        print(f"rows before join: {rows_before:,}")
        print(f"rows after join:  {rows_after:,}")
        print(f"dropped:          {dropped:,} ({dropped_pct:.2f}%)")
        if dropped_pct > DROP_FLAG_THRESHOLD_PCT:
            print(f"FLAG: join dropped more than {DROP_FLAG_THRESHOLD_PCT}% of rows")
            reasons.append(f"inner join dropped {dropped_pct:.2f}% of feature rows (> {DROP_FLAG_THRESHOLD_PCT}%)")

        label_col = find_column(merged_df, LABEL_NAME_CANDIDATES)
        if label_col:
            distribution = merged_df[label_col].value_counts(normalize=True) * 100
            print("label class distribution:")
            for label_value, pct in distribution.items():
                print(f"  {label_value}: {pct:.2f}%")
        else:
            print("FLAG: could not identify a label column in the merged data")
            reasons.append(f"no label column found in merged data (columns: {list(merged_df.columns)})")
    else:
        print("skipped: cannot join without a block ID column in both files")

    section("Null and zero-variance checks on merged data")
    if merged_df is not None:
        feature_columns = [
            column for column in merged_df.columns
            if column not in {features_id_col, labels_id_col, label_col}
        ]
        null_counts = merged_df[feature_columns].isnull().sum()
        print("null counts per feature column:")
        any_nulls = False
        for column, count in null_counts.items():
            if count > 0:
                any_nulls = True
                print(f"  FLAG {column}: {count:,} nulls")
            else:
                print(f"  {column}: 0 nulls")
        if any_nulls:
            reasons.append("one or more feature columns contain nulls after the join")

        zero_variance_columns = [
            column for column in feature_columns
            if merged_df[column].nunique(dropna=True) <= 1
        ]
        if zero_variance_columns:
            print(f"FLAG: zero-variance (constant) columns: {zero_variance_columns}")
            reasons.append(f"zero-variance columns found: {zero_variance_columns}")
        else:
            print("no zero-variance columns found")
    else:
        print("skipped: no merged data available")

    section("Verdict")
    if not reasons:
        print("READY FOR TRAINING")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path, index=False)
        print(f"Saved merged/labeled data to {output_path}")
        return True

    print("NOT READY — " + "; ".join(reasons))
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify features.csv aligns with anomaly_label.csv before training (no training performed)."
    )
    parser.add_argument("--features", default="data/features/features.csv")
    parser.add_argument("--labels", default="data/raw/hdfs/anomaly_label.csv")
    parser.add_argument("--output", default="data/features/features_labeled.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ready = verify(Path(args.features), Path(args.labels), Path(args.output))
    sys.exit(0 if ready else 1)
