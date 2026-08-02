"""MSST mel_band_roformer.py의 단일 스템 학습 버그 패치.

현 MSST 체크아웃은 멀티 스템 학습(active_stem_ids) 기능이 들어오면서
num_stems=1(우리 보컬 모델) 내부 손실 경로가 깨져 있다:
  - recon_audio가 [b, s, t] 3차원으로 squeeze된 뒤 'b n s t' 4차원 패턴으로
    rearrange → EinopsError ("expected 4 dims. Received 3-dim tensor")
  - target[:, stem_ids]가 3차원 타깃에서 스템이 아니라 채널 0을 선택
    (조용히 브로드캐스트되어 잘못된 l1 손실)

이 스크립트는 upstream(lucidrains) 원본과 같은 의미로 세 곳을 고친다.
멱등적이라 여러 번 실행해도 안전하며, setup_msst.sh 마지막에도 호출된다.

사용: python scripts/patch_msst.py
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = (ROOT / "third_party" / "Music-Source-Separation-Training"
          / "models" / "bs_roformer" / "mel_band_roformer.py")

REPLACEMENTS = [
    # (기존, 수정) — 단일 스템이면 타깃 전체를 그대로 사용 (채널 오선택 방지)
    (
        "        target_sel = target[:, stem_ids]",
        "        if recon_audio.ndim == 3:  # num_stems == 1 (single-target patch)\n"
        "            target_sel = target if target.ndim == 3 else target[:, 0]\n"
        "        else:\n"
        "            target_sel = target[:, stem_ids]",
    ),
    # 3/4차원 모두 동작하는 upstream 패턴으로 복원
    (
        "rearrange(recon_audio, 'b n s t -> (b n s) t')",
        "rearrange(recon_audio, '... s t -> (... s) t')",
    ),
    (
        "rearrange(target_sel, 'b n s t -> (b n s) t')",
        "rearrange(target_sel, '... s t -> (... s) t')",
    ),
]


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"파일이 없습니다: {TARGET}\n"
                         "먼저 bash scripts/setup_msst.sh 로 MSST를 설치하세요.")

    src = TARGET.read_text(encoding="utf-8")
    if "single-target patch" in src:
        print("이미 패치되어 있습니다 — 변경 없음.")
        return

    missing = [old for old, _ in REPLACEMENTS if old not in src]
    if missing:
        raise SystemExit(
            "패치 대상 코드를 찾지 못했습니다 (MSST 버전이 다르거나 이미 "
            f"수정됨):\n  {missing[0][:80]}...\n"
            "학습이 정상 동작하면 패치가 필요 없는 버전입니다.")

    for old, new in REPLACEMENTS:
        src = src.replace(old, new)

    backup = TARGET.with_suffix(".py.orig")
    if not backup.exists():
        backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(src, encoding="utf-8")

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        TARGET.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        raise SystemExit(f"패치 결과 문법 오류 — 원복했습니다:\n{exc}")

    print(f"패치 완료: {TARGET.relative_to(ROOT)}")
    print(f"(원본 백업: {backup.name})")


if __name__ == "__main__":
    main()
    sys.exit(0)
