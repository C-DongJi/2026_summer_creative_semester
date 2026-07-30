"""보고서 생성기 순수 계산 테스트."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_report import paired_deltas, render, set_means, verdict  # noqa: E402


def _rows(eval_set, values):
    return [
        {"eval_set": eval_set, "track": f"t{i}", "sdr_vocals": v, "sdr_inst": v + 1.0}
        for i, v in enumerate(values)
    ]


def test_set_means():
    rows = _rows("pop", [10.0, 12.0])
    means = set_means(rows)
    assert means["pop"]["vocals"] == pytest.approx(11.0)
    assert means["pop"]["inst"] == pytest.approx(12.0)
    assert means["pop"]["n"] == 2


def test_paired_deltas():
    base = _rows("pop", [10.0, 12.0])
    ft = _rows("pop", [10.5, 11.8])
    d = paired_deltas(base, ft)["pop"]
    assert d["mean"] == pytest.approx(0.15)
    assert d["wins"] == 1 and d["n"] == 2
    assert d["max"] == pytest.approx(0.5) and d["min"] == pytest.approx(-0.2)


def test_verdict_thresholds():
    assert "성공" in verdict(0.4, -0.05)[0]
    assert "경미한 망각" in verdict(0.4, -0.3)[0]
    assert "catastrophic" in verdict(0.4, -0.8)[0]
    assert "효과 없음" in verdict(0.05, 0.0)[0]
    assert "제한적" in verdict(0.2, 0.0)[0]
    assert "성공" in verdict(0.4, None)[0]  # 타 장르셋이 없어도 동작


def test_render_contains_key_sections():
    results = {
        "base": _rows("pop", [10.0]) + _rows("others", [9.0]),
        "ft": _rows("pop", [10.6]) + _rows("others", [8.95]),
    }
    md = render(results, "pop", {
        "name": "exp1", "train_data": "d", "epochs_lr": "5/1e-5",
        "results_path": "r.json",
    })
    assert "평가셋별 평균 SDR" in md
    assert "페어드 비교" in md
    assert "판정" in md
    assert "장르 특화 성공" in md  # pop +0.6, others -0.05 -> 성공
    assert "트랙별 상세" in md
