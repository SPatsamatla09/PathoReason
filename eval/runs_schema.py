from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class Track(str, Enum):
    A = "A"
    B = "B"


class MaskType(str, Enum):
    MEAN_FILL = "mean_fill"
    GAUSSIAN_BLUR = "gaussian_blur"
    BLACK_OCCLUSION = "black_occlusion"
    RANDOM_CONTROL = "random_control"


@dataclass
class RunRecord:
    image_id: str
    model: str
    track: Track
    prompt_id: str
    seed: int

    orig_label: str
    orig_conf: float
    annotator_agreement: Optional[int] = None

    feature_removed: Optional[str] = None
    mask_type: Optional[MaskType] = None
    new_label: Optional[str] = None
    new_conf: Optional[float] = None
    control_conf: Optional[float] = None
    sufficiency_conf: Optional[float] = None

    _VALID_PROMPT_PREFIXES = ("co_", "cte_", "etc_")

    def __post_init__(self):
        for name in ("orig_conf", "new_conf", "control_conf", "sufficiency_conf"):
            val = getattr(self, name)
            if val is not None and not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {val}")
        if self.annotator_agreement is not None and not (0 <= self.annotator_agreement <= 7):
            raise ValueError(f"annotator_agreement must be between 0 and 7, got {self.annotator_agreement}")
        if not self.prompt_id.startswith(self._VALID_PROMPT_PREFIXES):
            raise ValueError(
                f"prompt_id {self.prompt_id!r} doesn't match the frozen format "
                f"(co_/cte_/etc_ + p1/p2/p3)"
            )

    def to_json_line(self) -> str:
        d = asdict(self)
        d["track"] = self.track.value
        if self.mask_type is not None:
            d["mask_type"] = self.mask_type.value
        return json.dumps(d)


def append_run(record: RunRecord, path: Union[str, Path] = "results/runs.jsonl") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(record.to_json_line() + "\n")


def load_runs(path: Union[str, Path] = "results/runs.jsonl"):
    path = Path(path)
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    try:
        import pandas as pd
        return pd.DataFrame(records)
    except ImportError:
        return records


if __name__ == "__main__":
    r = RunRecord(
        image_id="001",
        model="gpt-4o",
        track=Track.B,
        prompt_id="cte_p1",
        seed=0,
        orig_label="Adenoma",
        orig_conf=0.987,
        annotator_agreement=6,
        feature_removed="gland formation",
        mask_type=MaskType.BLACK_OCCLUSION,
        new_label="Adenoma",
        new_conf=0.612,
        control_conf=0.95,
    )
    append_run(r, path="results/runs_test.jsonl")
    print(load_runs(path="results/runs_test.jsonl"))
