"""실험 결과 보고서 자동 생성기.

scripts/evaluate.py가 저장한 결과 JSON을 읽어 마크다운 보고서를 만든다:
요약 표, 페어드 비교, EVALUATION.md 기준의 자동 판정, 트랙별 부록,
청취 평가 기록란, 재현 명령까지 포함. 생성된 .md는 그대로 제출하거나
최종 보고서(한글/워드)에 붙여넣으면 된다.

사용 예:
    python scripts/make_report.py --results outputs/eval_pop_ft.json \
        --target-set pop --experiment-name pop-ft-01 \
        --train-data "MoisesDB pop 38곡" --epochs-lr "30 / 1e-5"
    # -> outputs/report_pop-ft-01.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------- 순수 계산 (테스트 대상) ----------

def set_means(rows: list[dict]) -> dict[str, dict]:
    """평가셋별 평균 {set: {vocals, inst, n}}."""
    out: dict[str, dict] = {}
    for r in rows:
        s = out.setdefault(r["eval_set"], {"vocals": 0.0, "inst": 0.0, "n": 0})
        s["vocals"] += r["sdr_vocals"]
        s["inst"] += r["sdr_inst"]
        s["n"] += 1
    for s in out.values():
        s["vocals"] /= s["n"]
        s["inst"] /= s["n"]
    return out


def paired_deltas(base_rows: list[dict], ft_rows: list[dict]) -> dict[str, dict]:
    """평가셋별 트랙 페어드 Δ(vocals) 통계."""
    out: dict[str, dict] = {}
    base = {(r["eval_set"], r["track"]): r for r in base_rows}
    for r in ft_rows:
        key = (r["eval_set"], r["track"])
        if key not in base:
            continue
        d = r["sdr_vocals"] - base[key]["sdr_vocals"]
        s = out.setdefault(r["eval_set"], {"deltas": []})
        s["deltas"].append((r["track"], d))
    for s in out.values():
        ds = [d for _, d in s["deltas"]]
        s["mean"] = sum(ds) / len(ds)
        s["wins"] = sum(d > 0 for d in ds)
        s["n"] = len(ds)
        s["max"] = max(ds)
        s["min"] = min(ds)
    return out


def verdict(target_delta: float, worst_other_delta: float | None) -> tuple[str, str]:
    """EVALUATION.md §5 기준 자동 판정 -> (라벨, 조치 안내)."""
    if target_delta < 0.1:
        return ("효과 없음", "데이터 양(40곡+)·품질(check_dataset) 점검, 에폭/lr 조정 후 재시도")
    if target_delta < 0.3:
        return ("△ 제한적 효과", "학습 데이터 확대 또는 에폭/lr 조정 검토. 향상 곡 비율로 보조 판단")
    if worst_other_delta is None or worst_other_delta >= -0.1:
        return ("✅ 장르 특화 성공", "채택. 보고서에 기록하고 Web UI 기본 장르 모델로 배포")
    if worst_other_delta >= -0.5:
        return ("⚠️ 경미한 망각", "장르 전용 모델로 명시해 사용하거나 lr 5e-6/에폭 축소로 재시도")
    return ("❌ catastrophic forgetting",
            "lr 5e-6, 에폭 축소, 학습셋에 타 장르 20% 혼합, LoRA 전환 순으로 재시도")


# ---------- 보고서 렌더링 ----------

def render(results: dict[str, list[dict]], target_set: str, meta: dict) -> str:
    models = list(results)
    lines: list[str] = []
    add = lines.append

    add(f"# 재학습 실험 보고서: {meta['name']}")
    add("")
    add(f"- 작성일: {date.today().isoformat()}")
    add(f"- 학습 데이터: {meta['train_data']}")
    add(f"- 에폭/lr: {meta['epochs_lr']}")
    add(f"- 비교 모델: {', '.join(f'`{m}`' for m in models)}")
    add(f"- 대상(재학습) 장르 평가셋: `{target_set}`")
    add("")

    add("## 1. 평가셋별 평균 SDR (dB)")
    add("")
    add("| 평가셋 | 곡 수 | " + " | ".join(f"{m} (vocals / inst)" for m in models) + " |")
    add("|---|---|" + "---|" * len(models))
    all_sets = sorted({r["eval_set"] for rows in results.values() for r in rows})
    for s in all_sets:
        cells = []
        n = 0
        for m in models:
            means = set_means(results[m])
            if s in means:
                n = means[s]["n"]
                cells.append(f"{means[s]['vocals']:+.2f} / {means[s]['inst']:+.2f}")
            else:
                cells.append("-")
        add(f"| {s} | {n} | " + " | ".join(cells) + " |")
    add("")

    if len(models) == 2:
        base_name, ft_name = models
        deltas = paired_deltas(results[base_name], results[ft_name])

        add(f"## 2. 페어드 비교: `{ft_name}` − `{base_name}` (vocals SDR)")
        add("")
        add("| 평가셋 | 평균 Δ | 향상 곡 | 최대 Δ | 최소 Δ |")
        add("|---|---|---|---|---|")
        for s in all_sets:
            if s not in deltas:
                continue
            d = deltas[s]
            add(f"| {s} | **{d['mean']:+.2f} dB** | {d['wins']}/{d['n']} | "
                f"{d['max']:+.2f} | {d['min']:+.2f} |")
        add("")

        if target_set in deltas:
            others = [d["mean"] for s, d in deltas.items() if s != target_set]
            worst_other = min(others) if others else None
            label, advice = verdict(deltas[target_set]["mean"], worst_other)
            add("## 3. 판정")
            add("")
            add(f"**{label}**")
            add("")
            add(f"- 근거: `{target_set}` 평균 Δ {deltas[target_set]['mean']:+.2f} dB"
                + (f", 타 장르 최저 Δ {worst_other:+.2f} dB" if worst_other is not None else ""))
            add(f"- 조치: {advice}")
            add(f"- 판정 기준: docs/EVALUATION.md §5")
            add("")

    add("## 4. 청취 평가 (수동 기록)")
    add("")
    add("| 곡 (평가셋) | base 소감 | 파인튜닝 소감 | 선호 |")
    add("|---|---|---|---|")
    add("| (예: TrackA (pop)) |  |  |  |")
    add("")

    add("## 5. 트랙별 상세 (부록)")
    add("")
    for m in models:
        add(f"### {m}")
        add("")
        add("| 평가셋 | 트랙 | vocals SDR | inst SDR |")
        add("|---|---|---|---|")
        for r in sorted(results[m], key=lambda x: (x["eval_set"], x["track"])):
            add(f"| {r['eval_set']} | {r['track']} | {r['sdr_vocals']:+.2f} | {r['sdr_inst']:+.2f} |")
        add("")

    add("## 6. 재현 정보")
    add("")
    add(f"- 결과 원본: `{meta['results_path']}`")
    add("- 실험 절차: docs/EXPERIMENT.md / 평가 설계: docs/EVALUATION.md")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 JSON -> 마크다운 보고서")
    parser.add_argument("--results", required=True, help="evaluate.py --output JSON 경로")
    parser.add_argument("--target-set", required=True, help="재학습 장르 평가셋 이름 (예: pop)")
    parser.add_argument("--experiment-name", default="finetune-exp")
    parser.add_argument("--train-data", default="(기입)", help="예: 'MoisesDB pop 38곡'")
    parser.add_argument("--epochs-lr", default="(기입)", help="예: '30 / 1e-5'")
    parser.add_argument("--output", default=None, help="기본: outputs/report_<이름>.md")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        results: dict[str, list[dict]] = json.load(f)
    if not results:
        raise SystemExit("결과 JSON이 비어 있습니다.")

    md = render(results, args.target_set, {
        "name": args.experiment_name,
        "train_data": args.train_data,
        "epochs_lr": args.epochs_lr,
        "results_path": args.results,
    })

    out = Path(args.output or f"outputs/report_{args.experiment_name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"보고서 생성: {out}")


if __name__ == "__main__":
    main()
