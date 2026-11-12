"""Per-block features from only the first k events of each block.

"First k events" means the first k lines for that block in file order, which
is arrival order: what a streaming detector would actually have seen when
the k-th event came in. The scan also counts how often a row's timestamp is
earlier than the row before it, so the report can say how far file order and
timestamp order disagree.

Memory: the 11M-row log is read in chunks with usecols, the low-cardinality
columns (and the timestamp, which repeats within a second) as categoricals so
each chunk encodes/parses only its distinct values, and the only state kept
between chunks is a fixed set of small per-block accumulators (counts capped
at k fit in uint8), never the events themselves.

A block with fewer than k events is described by all of its events. Its alert
can only fire once the block has finished, so its lead time is 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.category_encoder import DEFAULT_VOCAB_PATH, UNKNOWN, CategoryEncoder

DEFAULT_CLEANED_PATH = Path("data/processed/hdfs_cleaned.csv")
DEFAULT_PREFIX_DIR = Path("data/features/early_warning")
PREFIX_KS = (3, 5, 10, 20)
DEFAULT_CHUNKSIZE = 250_000

USECOLS = ["block_id", "datetime", "log_level", "component", "event_type", "block_size"]
WARN_LEVELS = ("WARN", "WARNING")
ERROR_LEVELS = ("ERROR", "FATAL", "CRITICAL")
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# Not features: needed only to compute lead time, and a leak if a model saw them.
TOTAL_EVENTS_COLUMN = "total_events"
PREFIX_END_COLUMN = "prefix_end_seconds"  # epoch seconds of the k-th event (when the alert fires)
BLOCK_END_COLUMN = "block_end_seconds"  # epoch seconds of the block's last event
BOOKKEEPING_COLUMNS = [TOTAL_EVENTS_COLUMN, PREFIX_END_COLUMN, BLOCK_END_COLUMN]
_NO_TIME = np.iinfo(np.int64).min

MISSING_FILE_HELP = """\
{path} not found. The early-warning features are built from the event-level
cleaned log, which a fresh clone does not have. Rebuild it with:

    python3 scripts/download_dataset.py     # HDFS.log + anomaly_label.csv from Hugging Face
    python3 scripts/build_hdfs_features.py  # parse -> clean -> features (15-30 min on the full log)
"""


def require_cleaned_log(path: str | Path = DEFAULT_CLEANED_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(MISSING_FILE_HELP.format(path=path))
    return path


class _Growable:
    """Per-block arrays that grow as new block ids appear mid-scan."""

    def __init__(self):
        self.capacity = 0
        self.arrays: dict[str, np.ndarray] = {}
        self.fill: dict[str, object] = {}

    def add(self, name: str, dtype, fill=0, width: int | None = None):
        shape = (self.capacity,) if width is None else (self.capacity, width)
        self.arrays[name] = np.full(shape, fill, dtype=dtype)
        self.fill[name] = fill

    def ensure(self, n_rows: int, widths: dict[str, int] | None = None):
        widths = widths or {}
        new_capacity = self.capacity
        while new_capacity < n_rows:
            new_capacity = max(1024, new_capacity * 2)
        for name, array in self.arrays.items():
            rows = max(new_capacity, array.shape[0])
            if array.ndim == 1:
                if rows > array.shape[0]:
                    grown = np.full((rows,), self.fill[name], dtype=array.dtype)
                    grown[: array.shape[0]] = array
                    self.arrays[name] = grown
            else:
                cols = max(widths.get(name, array.shape[1]), array.shape[1])
                if rows > array.shape[0] or cols > array.shape[1]:
                    grown = np.full((rows, cols), self.fill[name], dtype=array.dtype)
                    grown[: array.shape[0], : array.shape[1]] = array
                    self.arrays[name] = grown
        self.capacity = new_capacity

    def __getitem__(self, name):
        return self.arrays[name]


def _encode_categorical(encoder: CategoryEncoder, column: str, values: pd.Series) -> np.ndarray:
    """encoder.encode() over a categorical's distinct values only, then
    broadcast to the rows -- same codes, without a per-row string pass."""
    categories = pd.Series(values.cat.categories.astype("string").str.strip())
    category_codes = encoder.encode(column, categories).to_numpy()
    row_codes = values.cat.codes.to_numpy()
    if (row_codes < 0).any():
        # Only touch the vocabulary for UNKNOWN when a value is actually
        # missing: registering it eagerly would append to the shared vocab
        # file on every clean run.
        unknown = encoder.encode(column, pd.Series([UNKNOWN])).to_numpy()[0]
        return np.append(category_codes, unknown)[row_codes]
    return category_codes[row_codes]


def feature_columns(event_types: list[str]) -> list[str]:
    """Model input columns, in a fixed order (count columns follow vocab code order)."""
    return [
        "n_events", "elapsed_seconds", "distinct_event_types", "distinct_components",
        "warn_count", "error_count", "block_size_max",
        "first_component_encoded", "first_event_type_encoded",
        "dominant_component_encoded", "dominant_event_type_encoded",
    ] + [f"count_{event_type}" for event_type in event_types]


def build_prefix_features(
    cleaned_path: str | Path = DEFAULT_CLEANED_PATH,
    ks=PREFIX_KS,
    chunksize: int = DEFAULT_CHUNKSIZE,
    vocab_path: str | Path = DEFAULT_VOCAB_PATH,
    verbose: bool = False,
) -> tuple[dict[int, pd.DataFrame], dict]:
    """Returns ({k: one row per block}, scan stats). Each frame has block_id,
    the feature_columns(), and BOOKKEEPING_COLUMNS (lead-time bookkeeping only):
    total_events, prefix_end_seconds (latest time among the first k events,
    i.e. the k-th event's time when the log is in time order) and
    block_end_seconds (the block's last event). Times are epoch seconds; NaN
    when a block has no parsable timestamp."""
    path = require_cleaned_log(cleaned_path)
    ks = sorted(set(int(k) for k in ks))
    if not ks or ks[0] < 1 or ks[-1] > 255:
        raise ValueError("every k must be between 1 and 255")
    k_max = ks[-1]

    encoder = CategoryEncoder.load(vocab_path)
    vocab_before = encoder.summary()

    block_codes: dict[str, int] = {}
    block_ids: list[str] = []
    state = _Growable()
    state.add("seen", np.int64)  # events per block so far (whole block, not capped)
    state.add("t_last", np.int64, fill=_NO_TIME)  # latest event time over the whole block
    state.add("first_component", np.int16, fill=-1)
    state.add("first_event_type", np.int16, fill=-1)
    for k in ks:
        state.add(f"type_counts_{k}", np.uint8, width=1)
        state.add(f"component_counts_{k}", np.uint8, width=1)
        state.add(f"warn_{k}", np.uint8)
        state.add(f"error_{k}", np.uint8)
        state.add(f"block_size_max_{k}", np.float64, fill=0.0)
        state.add(f"t_min_{k}", np.int64, fill=np.iinfo(np.int64).max)
        state.add(f"t_max_{k}", np.int64, fill=np.iinfo(np.int64).min)

    stats = {"rows": 0, "rows_skipped_no_block_id": 0, "rows_out_of_time_order": 0, "chunks": 0}
    previous_time = None

    reader = pd.read_csv(
        path, usecols=USECOLS, chunksize=chunksize,
        dtype={"block_id": "string", "datetime": "category", "log_level": "category", "component": "category",
               "event_type": "category", "block_size": "float64"},
    )
    for chunk in reader:
        stats["chunks"] += 1
        stats["rows"] += len(chunk)
        valid = chunk["block_id"].notna() & (chunk["block_id"].str.strip() != "")
        stats["rows_skipped_no_block_id"] += int((~valid).sum())
        chunk = chunk[valid]
        if chunk.empty:
            continue

        stamps = pd.to_datetime(chunk["datetime"].cat.categories, format=DATETIME_FORMAT, errors="coerce")
        stamp_seconds = np.where(stamps.isna(), np.iinfo(np.int64).min, stamps.asi8 // 10**9)  # NaT never sets min/max
        seconds = np.append(stamp_seconds, np.iinfo(np.int64).min)[chunk["datetime"].cat.codes.to_numpy()]
        if previous_time is not None and len(seconds):
            stats["rows_out_of_time_order"] += int(seconds[0] < previous_time)
        stats["rows_out_of_time_order"] += int((np.diff(seconds) < 0).sum())
        previous_time = seconds[-1]

        ids = chunk["block_id"].str.strip()
        for block_id in pd.unique(ids):
            if block_id not in block_codes:
                block_codes[block_id] = len(block_ids)
                block_ids.append(block_id)
        codes = ids.map(block_codes).to_numpy(dtype=np.int64)

        event_type = _encode_categorical(encoder, "event_type", chunk["event_type"])
        component = _encode_categorical(encoder, "component", chunk["component"])
        n_types, n_components = len(encoder.vocab["event_type"]), len(encoder.vocab["component"])
        widths = {}
        for k in ks:
            widths[f"type_counts_{k}"] = n_types
            widths[f"component_counts_{k}"] = n_components
        state.ensure(len(block_ids), widths)

        levels = pd.Series(chunk["log_level"].cat.categories.astype("string").str.strip().str.upper())
        level_code = chunk["log_level"].cat.codes.to_numpy()
        warn = np.append(levels.isin(WARN_LEVELS).to_numpy(), False)[level_code]
        error = np.append(levels.isin(ERROR_LEVELS).to_numpy(), False)[level_code]
        block_size = chunk["block_size"].fillna(0.0).to_numpy()

        # Position of each row within its block, continuing across chunks.
        position = state["seen"][codes] + pd.Series(codes).groupby(codes).cumcount().to_numpy()
        np.add.at(state["seen"], codes, 1)

        has_time = seconds != _NO_TIME
        np.maximum.at(state["t_last"], codes[has_time], seconds[has_time])

        first = position == 0
        state["first_component"][codes[first]] = component[first]
        state["first_event_type"][codes[first]] = event_type[first]

        in_prefix_max = position < k_max
        if not in_prefix_max.any():
            continue
        for k in ks:
            m = position < k
            if not m.any():
                continue
            c = codes[m]
            np.add.at(state[f"type_counts_{k}"], (c, event_type[m]), 1)
            np.add.at(state[f"component_counts_{k}"], (c, component[m]), 1)
            np.add.at(state[f"warn_{k}"], c, warn[m].astype(np.uint8))
            np.add.at(state[f"error_{k}"], c, error[m].astype(np.uint8))
            np.maximum.at(state[f"block_size_max_{k}"], c, block_size[m])
            valid_time = seconds[m] != np.iinfo(np.int64).min
            np.minimum.at(state[f"t_min_{k}"], c[valid_time], seconds[m][valid_time])
            np.maximum.at(state[f"t_max_{k}"], c[valid_time], seconds[m][valid_time])

        if verbose:
            print(f"  chunk {stats['chunks']}: {stats['rows']:,} rows, {len(block_ids):,} blocks")

    if encoder.summary() != vocab_before:
        # Same contract as the feature extractor: persist appended values so
        # the codes these models learned stay assigned.
        encoder.save(vocab_path)

    n_blocks = len(block_ids)
    event_types = [name for name, _ in sorted(encoder.vocab["event_type"].items(), key=lambda item: item[1])]
    columns = feature_columns(event_types)
    frames = {}
    for k in ks:
        # int32 throughout: four 575k-row frames stay ~50 MB each.
        type_counts = state[f"type_counts_{k}"][:n_blocks].astype(np.int32)
        component_counts = state[f"component_counts_{k}"][:n_blocks]
        # Pad to the final vocab width (a code first seen late leaves earlier
        # arrays narrower only if no block in that k ever had it -- not a
        # problem, but the frame must still have every count column).
        if type_counts.shape[1] < len(event_types):
            type_counts = np.pad(type_counts, ((0, 0), (0, len(event_types) - type_counts.shape[1])))
        t_min, t_max = state[f"t_min_{k}"][:n_blocks], state[f"t_max_{k}"][:n_blocks]
        has_time = t_max >= t_min
        data = {
            "block_id": block_ids,
            "n_events": type_counts.sum(axis=1, dtype=np.int32),
            "elapsed_seconds": np.where(has_time, t_max - t_min, 0).astype(np.int32),
            "distinct_event_types": (type_counts > 0).sum(axis=1, dtype=np.int32),
            "distinct_components": (component_counts > 0).sum(axis=1, dtype=np.int32),
            "warn_count": state[f"warn_{k}"][:n_blocks].astype(np.int32),
            "error_count": state[f"error_{k}"][:n_blocks].astype(np.int32),
            "block_size_max": state[f"block_size_max_{k}"][:n_blocks],
            "first_component_encoded": state["first_component"][:n_blocks].astype(np.int32),
            "first_event_type_encoded": state["first_event_type"][:n_blocks].astype(np.int32),
            # argmax breaks ties toward the smallest code, as the feature
            # extractor's dominant value does.
            "dominant_component_encoded": component_counts.argmax(axis=1).astype(np.int32),
            "dominant_event_type_encoded": type_counts.argmax(axis=1).astype(np.int32),
        }
        for code, event_type in enumerate(event_types):
            data[f"count_{event_type}"] = type_counts[:, code]
        data[TOTAL_EVENTS_COLUMN] = state["seen"][:n_blocks].astype(np.int32)
        data[PREFIX_END_COLUMN] = np.where(has_time, t_max, np.nan)
        t_last = state["t_last"][:n_blocks]
        data[BLOCK_END_COLUMN] = np.where(t_last != _NO_TIME, t_last, np.nan)
        frames[k] = pd.DataFrame(data)[["block_id"] + columns + BOOKKEEPING_COLUMNS]

    stats["blocks"] = n_blocks
    stats["events"] = int(state["seen"][:n_blocks].sum())
    return frames, stats
