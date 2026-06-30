"""설정 로딩 유틸.

YAML 설정 파일을 읽어 점(dot) 접근이 가능한 dict로 반환한다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict를 속성처럼 접근할 수 있게 감싼 헬퍼 (cfg.audio.sample_rate)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict):
            return Config(value)
        return value


def load_config(path: str | Path) -> Config:
    """YAML 설정 파일을 로드한다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)
