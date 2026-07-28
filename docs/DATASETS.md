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

---

# 추가 후보 데이터셋 (2026-07-28 웹 리서치)

## Kim 체크포인트는 뭘로 학습됐나

**비공개·비문서화.** HF 모델 카드는 비어 있고, GitHub README와 커뮤니티 가이드(deton24)에
따르면 **Kimberley Jensen + aufr33/Anjok(UVR 개발진) + Bas Curtiz가 모은 사설 데이터셋**으로
학습됐으며 구성은 공개된 적이 없다. 정황상(UVR 계열의 수집 방식) **MUSDB18-HQ와 상용
아카펠라/공식 반주 쌍이 포함됐을 가능성이 높다**(추정). MoisesDB/MedleyDB 포함 여부는
확인 불가.

→ 실무 지침: **MUSDB18-HQ/musdb-XL/DSD100은 학습 데이터에서 제외**(중복 위험 최대 +
MUSDB 훈련셋 100곡 중 46곡은 MedleyDB와도 겹침). MUSDB18-HQ **test 50곡은 학습에 안
쓰고 평가 전용**으로만 사용. MedleyDB 사용 시 MUSDB18에 포함된 46곡을 제외하면 더 안전.

## 추가 후보 요약 (기존 MoisesDB/MedleyDB 외)

| 데이터셋 | 내용 | 2-stem | 라이선스 | 비고 |
|---|---|---|---|---|
| **GTSinger** (2024-09) | 보컬 단독 80.6h, 9개 언어(한국어 포함), 48kHz | 합성 믹싱용 | CC BY-NC-SA 4.0 | Kim 공개(2024-08) 이후 출시 → **겹침 거의 0** |
| **AIHub 가이드보컬 #473 / 다화자 가창 #465** | K-pop 가이드보컬 157h·4천곡 / 1,500곡 | 합성 믹싱용 | AIHub 약관(한국인, 승인) | **K-pop 도메인 적응 최적**, 서구권 모델 미사용 |
| **ccMixter (Liutkus)** | 50곡 (music, voice, mix) 트리플 | **직접** | CC 계열 | 5.2GB 직링크, 게이팅 없음 |
| **RawStems** (2025) | Cambridge-MT 기반 578곡/354h 원본 스템 | 보컬 그룹 + 나머지 합 | 교육용(Cambridge-MT 승계) | 214GB HF. 마이크 블리드 선별 필요, DSD100/MUSDB 곡 제외 필수 |
| **DAMP-VSEP** | 카라오케 11,494곡 (반주,보컬,믹스) | **직접** | Smule 연구 라이선스(신청) | 폰 녹음 품질 — 필터링 후 강건성 데이터로 |
| SingStyle111 | 보컬 단독 12.8h | 합성 믹싱용 | **CC BY 4.0** (가장 깨끗) | |
| CSD (어린이 동요) | 한국어/영어 100곡, 보컬 단독 | 합성 믹싱용 | CC BY-NC-SA | Zenodo 공개 |
| Slakh2100 | 반주 전용 145h (보컬 없음) | vocals=무음 예제 | CC BY 4.0 | 보컬 잔향 오탐 억제용 네거티브 |
| ~~MUSDB18-HQ / DSD100~~ | — | — | — | **학습 제외** (Kim 중복 위험) |
| ~~SDX23 corrupted~~ | — | — | — | MoisesDB 203곡 부분집합 = 기존과 중복 |

## 합성 믹싱 전략 (보컬 단독 + 반주 단독 → 학습 쌍)

MSST **dataset type 2/3**는 스템별 폴더/CSV에서 무작위 조합으로 믹스를 만들어 학습을
지원한다 → GTSinger·AIHub 보컬을 `vocals/`에, Slakh·기존 반주를 `other/`에 넣으면 바로
2-stem 학습 가능. (vocals=무음 + Slakh) 쌍은 반주에서 보컬을 오탐하는 문제를 줄인다.
단, 무반주 스튜디오 보컬은 실제 믹스의 보컬 체인(리버브/컴프레서)과 달라 리버브·EQ 증강을
곁들이고 합성 쌍 비중을 배치의 일부로 제한할 것.

## 권장 우선순위

1. **GTSinger + SingStyle111 + CSD** (보컬) × **Slakh2100 + 기존 반주** (합성 믹싱) — 겹침 최소·라이선스 깨끗·즉시 다운로드
2. **AIHub #473/#465** — K-pop 특화 목표(설계서의 "장르 특화")에 정확히 부합, 한국인 승인 필요
3. **ccMixter** — 실제 믹스 트리플 50곡, 가장 간단한 추가
4. **RawStems** — 대량 실측 멀티트랙, 블리드 선별 + MUSDB 중복 제거 전제
5. **DAMP-VSEP** — 규모는 최대, 품질 필터링 필수
