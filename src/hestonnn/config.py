"""Config loader. Every run is driven by a YAML file; nothing is hand-tuned in code."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class Config:
    raw: dict
    data_root: Path

    @classmethod
    def load(cls, path: str | Path, data_root: str | Path | None = None) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        root = Path(data_root) if data_root else Path(raw["data_root"])
        return cls(raw=raw, data_root=root)

    # convenience accessors -------------------------------------------------
    def src(self, key: str) -> Path:
        return self.data_root / self.raw["sources"][key]

    @property
    def close_window(self) -> tuple[str, str]:
        cw = self.raw["close_window"]
        return cw["start"], cw["end"]

    @property
    def tv_threshold(self) -> float:
        return float(self.raw["expiry"]["time_value_threshold"])

    @property
    def surf(self) -> dict:
        return self.raw["surface"]

    def out(self, key: str) -> Path:
        # outputs are written relative to the project dir, not the data root
        return Path(self.raw["output"][key])
