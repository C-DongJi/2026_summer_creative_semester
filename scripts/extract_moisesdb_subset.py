"""MoisesDB zip에서 원하는 장르 트랙만 선별 추출 (전체 압축 해제 불필요).

80GB 아카이브를 통째로 풀면 디스크가 160GB 필요하지만, 이 스크립트는
zip 내부의 트랙별 data.json(수 KB)만 먼저 읽어 장르를 확인한 뒤
해당 트랙 폴더만 추출한다 (pop 기준 최종 수 GB).

사용 예:
    # 1) 장르 분포만 미리 보기 (추출 없음, 수십 초)
    python scripts/extract_moisesdb_subset.py --zip moisesdb.zip --list-genres

    # 2) 팝만 추출
    python scripts/extract_moisesdb_subset.py --zip moisesdb.zip \
        --genres pop --out /data/moisesdb_pop

    # 3) 평가셋 B용 타 장르 추출
    python scripts/extract_moisesdb_subset.py --zip moisesdb.zip \
        --genres rock jazz electronic --out /data/moisesdb_others

추출 결과 폴더는 그대로 prepare_dataset.py --src 로 사용할 수 있다.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest_common import matches_genre  # noqa: E402


def scan_tracks(zf: zipfile.ZipFile) -> dict[str, str]:
    """zip 안의 트랙 폴더 prefix -> 장르 문자열. (data.json만 읽음 — 빠름)"""
    tracks: dict[str, str] = {}
    for name in zf.namelist():
        if not name.endswith("/data.json"):
            continue
        prefix = name[: -len("data.json")]
        try:
            meta = json.loads(zf.read(name))
            tracks[prefix] = str(meta.get("genre", "") or "")
        except (json.JSONDecodeError, KeyError):
            tracks[prefix] = ""
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="MoisesDB zip 장르 선별 추출")
    parser.add_argument("--zip", required=True, help="moisesdb 아카이브(.zip) 경로")
    parser.add_argument("--genres", nargs="+", default=None,
                        help="추출할 장르 (부분 일치). 예: --genres pop")
    parser.add_argument("--out", default=None, help="추출 대상 디렉토리")
    parser.add_argument("--list-genres", action="store_true",
                        help="장르별 트랙 수만 출력하고 종료")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"파일이 없습니다: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        print("트랙 메타데이터 스캔 중 (data.json만 읽음)...")
        tracks = scan_tracks(zf)
        print(f"총 {len(tracks)}개 트랙 발견")

        counts = Counter(g.lower() or "(장르 없음)" for g in tracks.values())
        print("\n장르 분포:")
        for genre, n in counts.most_common():
            print(f"  {genre:<25} {n:>4}곡")

        if args.list_genres:
            return
        if not args.genres or not args.out:
            raise SystemExit("\n추출하려면 --genres 와 --out 을 지정하세요.")

        selected = [p for p, g in tracks.items() if matches_genre(g, args.genres)]
        if not selected:
            raise SystemExit(f"일치하는 트랙이 없습니다: {args.genres}")
        print(f"\n선택된 트랙: {len(selected)}곡 -> {args.out} 추출 시작")

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        prefixes = tuple(selected)
        members = [n for n in zf.namelist() if n.startswith(prefixes)]
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, out)
            if i % 200 == 0 or i == total:
                print(f"  ... {i}/{total} 파일")

    # zip에 루트 폴더(예: moisesdb_v0.1/)가 있으면 --src는 그 안쪽을 가리켜야 함
    top = [p for p in out.iterdir() if p.is_dir()]
    src_hint = top[0] if len(top) == 1 else out
    print(f"\n완료. 다음 단계:\n"
          f"  python scripts/prepare_dataset.py --dataset moisesdb "
          f"--src {src_hint} --genres {' '.join(args.genres)}")


if __name__ == "__main__":
    main()
