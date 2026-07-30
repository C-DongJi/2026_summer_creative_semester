"""웹 재학습 탭 테스트용 (mix, inst) 쌍 생성기.

MUSDB18 공식 7초 프리뷰 샘플(오픈 액세스, 자동 다운로드 ~140MB)에서
보컬이 뚜렷한 트랙을 골라 Web UI 재학습 탭의 파일명 규칙에 맞는
<곡명>_mix.wav / <곡명>_inst.wav 쌍으로 내보낸다.

주의: MUSDB는 학습 제외 원칙이지만 이 쌍은 '기능 테스트'용이다.
     실제 품질 개선용 파인튜닝은 MoisesDB 등을 사용할 것 (docs/DATASETS.md).

사용 예:
    pip install musdb   # 1회
    python scripts/make_demo_pairs.py --count 3
    # -> outputs/demo_pairs/ 에 쌍 파일 생성, 재학습 탭에 업로드
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="재학습 테스트용 (mix, inst) 쌍 생성")
    parser.add_argument("--count", type=int, default=3, help="생성할 곡 수")
    parser.add_argument("--out", default="outputs/demo_pairs", help="출력 디렉토리")
    parser.add_argument("--musdb-root", default="data/musdb_sample",
                        help="MUSDB 샘플 저장 위치 (없으면 자동 다운로드)")
    args = parser.parse_args()

    try:
        import musdb
    except ImportError:
        raise SystemExit("musdb 패키지가 없습니다. 먼저: pip install musdb")

    db = musdb.DB(root=args.musdb_root, download=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 보컬 에너지가 큰 트랙 우선
    scored = []
    for t in db:
        v = t.targets["vocals"].audio
        scored.append((float(np.sqrt((v ** 2).mean())), t))
    scored.sort(key=lambda x: -x[0])

    for rms, t in scored[: args.count]:
        name = re.sub(r"[^0-9A-Za-z가-힣_-]", "_", t.name)[:40]
        mix = t.audio
        inst = t.targets["accompaniment"].audio  # 보컬 제외 전체 합
        sf.write(out / f"{name}_mix.wav", mix, t.rate)
        sf.write(out / f"{name}_inst.wav", inst, t.rate)
        print(f"  {name}  (보컬 RMS {rms:.3f}, {mix.shape[0]/t.rate:.1f}s)")

    print(f"\n{args.count}곡 쌍 생성 완료: {out}")
    print("Web UI '재학습' 탭에서 이 파일들을 전부 업로드하면 됩니다.")


if __name__ == "__main__":
    main()
