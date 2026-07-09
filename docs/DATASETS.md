# 공개 데이터셋으로 파인튜닝하기 (MoisesDB · MedleyDB)

## 제약 조건과의 관계 (중요)

본 프로젝트의 파인튜닝 데이터 제약은 **(원본 Mix, 반주) 2-stem** — 악기별 멀티 스템
supervision 사용 불가다. MoisesDB·MedleyDB는 원래 **악기별 멀티 스템** 데이터셋이지만,
아래처럼 **2-stem으로 접어서(collapse)** 쓰면 제약을 지킨다:

```
vocals  = 보컬 스템(들)의 합
other   = 보컬 이외 모든 스템의 합
mixture = vocals + other            (스템 합으로 자기정합적으로 정의)
```

→ 모델은 여전히 **보컬/반주만** 학습한다(드럼·베이스 등 개별 스템을 구분하지 않음).
멀티 스템 정보는 `other` 하나로 합쳐지며 버려진다. 즉 데이터셋이 멀티 스템이어도
**학습 신호는 2-stem**이므로 제약을 위반하지 않는다.

## ⚠️ 라이선스 주의

| 데이터셋 | 라이선스 | 의미 |
|---|---|---|
| MoisesDB | CC BY-NC-SA 4.0 | **비상업** 연구용만 |
| MedleyDB | CC BY-NC-SA | **비상업** 연구용만 |

두 데이터셋 모두 **비상업(NC)** 이다. 이 데이터로 파인튜닝한 가중치는 상업적 이용이
제한될 수 있다. (베이스 모델 Kim 체크포인트는 MIT지만, NC 데이터로 학습한 파생물은
데이터 라이선스의 영향을 받는다.) 창의학기제 연구·데모 용도로는 문제없다.

## 데이터셋 구조 (참고)

### MoisesDB (240곡, 44.1kHz 스테레오)
- 11개 top-level 스템: `vocals`, `bass`, `drums`, `guitar`, `piano`, `other`,
  `other_keys`, `other_plucked`, `percussion`, `bowed_strings`, `wind`
- 각 스템은 여러 소스 wav의 합. 공식 패키지가 합산·정렬을 처리.
- `vocals` 없는 트랙은 자동 스킵.

### MedleyDB (V1 122곡 + 2.0까지 196곡, 44.1kHz, 스템 스테레오)
- 트랙의 약 40~45%가 순수 반주(instrumental) → `is_instrumental`로 자동 스킵.
- 보컬 라벨(taxonomy `voices`): `male/female singer`, `vocalists`, `choir`,
  `male/female rapper`, `beatboxing`, `male/female screamer`
  (기본값은 이 '가창' 집합. `--include-speech`로 `speaker`/`crowd` 추가 가능)
- `has_bleed`(스템 간 누화) 트랙은 기본 제외(`--keep-bleed`로 포함). V1의 25곡이 해당.
- mixture는 마스터링된 `_MIX.wav`가 아니라 **스템 합**으로 정의(자기정합적).

## 설치

```bash
conda activate changuihakgi

# MoisesDB 패키지 (PyPI 미배포 → git)
pip install git+https://github.com/moises-ai/moises-db.git

# MedleyDB 패키지
pip install medleydb
```

데이터 자체는 별도로 받아야 한다:
- **MoisesDB**: https://music.ai/research/ (datasets) 에서 다운로드 후 압축 해제.
- **MedleyDB**: Zenodo 승인 요청 — V1 `zenodo.org/record/1649325`, 2.0 `zenodo.org/records/1715175`.
  받은 뒤 `export MEDLEYDB_PATH=/path/to/MedleyDB` (하위에 `Audio/` 폴더).

## 사용법

### 1) 2-stem 학습셋으로 변환

```bash
# MoisesDB
python scripts/prepare_dataset.py --dataset moisesdb \
    --src /data/moisesdb_v0.1 --out data/processed --valid-dir data/valid

# MedleyDB (MEDLEYDB_PATH 설정 후)
python scripts/prepare_dataset.py --dataset medleydb \
    --versions V1 V2 --out data/processed --valid-dir data/valid
```

결과: `data/processed/<dataset>__<track>/{vocals.wav, other.wav}` (32-bit float),
그리고 `--holdout-frac`(기본 0.1) 비율만큼 `data/valid`로 결정적 분리.
두 데이터셋을 같은 `--out`에 넣으면 자연스럽게 합쳐진다(트랙명 prefix로 구분).

### 2) 데이터셋 점검 (필수)

변환 후 반드시 점검해서 학습에 문제될 트랙을 걸러낸다:

```bash
python scripts/check_dataset.py --dir data/processed
python scripts/check_dataset.py --dir data/valid --verbose
```

점검 항목:
- `missing_stem` — vocals/other 누락
- `len_mismatch` / `sr_mismatch` / `ch_mismatch` — 스템 간 불일치
- `nan_or_inf` — 비정상 샘플
- `silent_vocals` — vocals가 사실상 무음(반주 트랙 혼입 의심)
- `mix_clip` — vocals+other 피크가 1.0 초과(클리핑)

이상이 있으면 종료코드 1. 통계(트랙 수·총 시간·길이 분포)도 함께 출력.

### 3) 파인튜닝

점검을 통과한 `data/processed`를 그대로 학습에 사용한다 — [PIPELINE_DESIGN.md](PIPELINE_DESIGN.md) Mode 2 참고.
Dataset/DataLoader는 `vocals.wav`+`other.wav`를 읽고 `mixture = vocals + other`로 재구성한다.

## 데이터 규모 참고

커뮤니티 선례상 유의미한 파인튜닝은 **~40곡**부터, 견고하려면 **170곡+**.
MoisesDB(240곡) + MedleyDB 보컬 트랙(V1 기준 ~70곡, 누화 제외 시 더 적음)을
합치면 충분한 규모를 확보할 수 있다.
