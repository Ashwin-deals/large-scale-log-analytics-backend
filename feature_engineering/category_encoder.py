"""Stable integer codes for categorical log fields.

The encoding used to be derived per build:

    {value: index for index, value in enumerate(sorted(values.unique()))}

which meant a code depended on whichever rows happened to be in that slice.
Rebuilding features over a different portion of the log silently renumbered
every component and IP, so a saved model's learned thresholds no longer
referred to the same categories — predictions became meaningless without any
error being raised.

The vocabulary is now persisted. Values keep the code they were first given,
and unseen values are appended, so codes mean the same thing across builds and
a model stays valid against features rebuilt later.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

DEFAULT_VOCAB_PATH = Path("data/features/category_encoders.json")
ENCODED_COLUMNS = ["component", "event_type", "source_ip", "destination_ip"]
UNKNOWN = "UNKNOWN"


class CategoryEncoder:
    """Append-only {value: code} maps, one per categorical column."""

    def __init__(self, vocab: dict[str, dict[str, int]] | None = None):
        self.vocab = vocab or {}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_VOCAB_PATH) -> "CategoryEncoder":
        path = Path(path)
        if not path.exists():
            return cls({})
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: str | Path = DEFAULT_VOCAB_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.vocab, indent=2, sort_keys=True), encoding="utf-8")

    def encode(self, column: str, series: pd.Series) -> pd.Series:
        """Map `series` to codes, registering any values not seen before.

        New values are appended in sorted order after the highest existing
        code, so a given build is reproducible and earlier codes never shift.
        """
        values = series.astype("string").fillna(UNKNOWN)
        mapping = self.vocab.setdefault(column, {})

        unseen = sorted(set(values.unique()) - set(mapping))
        next_code = max(mapping.values(), default=-1) + 1
        for value in unseen:
            mapping[value] = next_code
            next_code += 1

        return values.map(mapping).astype("int64")

    def fingerprint(self) -> str:
        """Short digest identifying this vocabulary.

        Stored on the model at fit time so inference can tell whether the
        features it is given were encoded the same way, instead of silently
        scoring against renumbered categories.
        """
        payload = json.dumps(self.vocab, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def summary(self) -> dict[str, int]:
        return {column: len(mapping) for column, mapping in sorted(self.vocab.items())}

    def conflicts_with(self, trained_vocab: dict[str, dict[str, int]]) -> list[str]:
        """Codes in `trained_vocab` that this vocabulary assigns differently.

        Growth is fine — an uploaded log routinely contains a host the training
        data never saw, and appending it leaves every existing code untouched.
        What breaks a model is a value being *renumbered*, because its learned
        splits then refer to a different category. Only that is reported.
        """
        conflicts = []
        for column, mapping in (trained_vocab or {}).items():
            current = self.vocab.get(column, {})
            for value, code in mapping.items():
                now = current.get(value)
                if now is not None and now != code:
                    conflicts.append(f"{column}[{value}]: trained as {code}, now {now}")
        return conflicts
