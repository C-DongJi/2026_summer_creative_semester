# 재학습 효과 평가 실험 설계

파인튜닝으로 결과가 얼마나 좋아졌는지(그리고 나빠진 곳은 없는지)를 확인하는 실험 절차.
핵심 질문은 두 개다:

- **Q1. 재학습한 장르에서 분리 품질이 올랐는가?** (목표 효과)
- **Q2. 다른 장르에서 품질이 떨어지지 않았는가?** (부작용: catastrophic forgetting)

둘을 같이 봐야 한다. Q1만 보면 "특정 장르에 과적합해 나머지를 망친 모델"을
성공으로 오판할 수 있다.

## 1. 지표: 글로벌 SDR (dB)

`SDR = 10·log10(Σref² / Σ(ref−est)²)` — 트랙 전체에 대해 계산 (uSDR 계열,
MDX 챌린지·MVSep 리더보드·MSST와 같은 정의, [src/utils/metrics.py](../src/utils/metrics.py)).
vocals와 instrumental 각각 계산하며, 높을수록 좋다. 참고로 공개 상위 모델들의
multisong vocals SDR이 11~12 dB 수준이다.

## 2. 평가셋 구성 (3종)

모두 MSST 레이아웃(<곡>/{vocals.wav, other.wav})이며, **학습에 쓴 트랙은 절대
포함하지 않는다** (누설 방지 — 아래 3절).

| 평가셋 | 구성 | 답하는 질문 |
|---|---|---|
| **A. 재학습 장르 홀드아웃** | 재학습 장르(예: pop)에서 학습에 안 쓴 트랙 (prepare_dataset가 자동 분리한 `data/valid` 중 해당 장르) | Q1 |
| **B. 타 장르셋** | MoisesDB의 다른 장르들(rock, jazz, electronic 등)에서 뽑은 트랙 — 학습에 전혀 안 쓴 것 | Q2 |
| **C. (권장) 외부 표준셋** | MUSDB18-HQ **test** 50곡 (Zenodo 공개). 우리 학습에 안 쓰므로 평가 전용으로 적합하고, 공개 리더보드 수치와 비교 가능 | Q2 + 외부 비교 |

만드는 법 (예: pop 재학습 실험):

```bash
# A+학습셋: pop만 변환 — valid로 분리된 것이 평가셋 A
python scripts/prepare_dataset.py --dataset moisesdb --genres pop \
    --out data/train_pop --valid-dir data/eval/pop

# B: 타 장르를 별도 폴더로 (학습에 쓰지 않음 — 전부 평가용)
python scripts/prepare_dataset.py --dataset moisesdb --genres rock jazz electronic \
    --out data/eval/others --valid-dir data/eval/others --holdout-frac 0

# C: MUSDB18-HQ test를 받아 <곡>/{vocals.wav, other.wav}로 변환
#    (vocals = vocals.wav 그대로, other = bass+drums+other 합 — 필요 시 스크립트 요청)
```

평가셋 규모 가이드: A ≥ 5곡, B ≥ 10곡(장르당 3곡 이상 섞이게), C는 50곡 전부 또는
시간이 없으면 `--max-tracks 15`.

## 3. 누설(leakage) 점검 — 실험 전 필수

- 평가셋 A/B의 트랙 폴더명이 학습셋(`data/train_pop`)에 없는지 확인:
  ```bash
  comm -12 <(ls data/train_pop | sort) <(ls data/eval/pop data/eval/others | sort)
  # 출력이 비어 있어야 함
  ```
- Kim 베이스 모델의 학습 데이터는 비공개이므로 **절대 SDR보다 base 대비 Δ(변화량)로
  해석**한다 ([DATASETS.md](DATASETS.md)의 Kim 학습 데이터 절 참고).
- MUSDB18-HQ는 **train을 우리 학습에 쓰지 않는다는 전제**로 C가 성립한다 (기존 원칙 유지).

## 4. 실행 프로토콜

같은 config(청크 8초, overlap 0.75, precision auto)로 base와 finetuned를 **같은
평가셋에서** 돌린다. 분리 파이프라인은 결정적이므로 반복 실행 불필요.

```bash
python scripts/evaluate.py \
  --eval-set pop=data/eval/pop \
  --eval-set others=data/eval/others \
  --model base=models/checkpoints/MelBandRoformer.ckpt \
  --model pop_ft=models/checkpoints/finetune_pop/<체크포인트>.ckpt \
  --output outputs/eval_pop_ft.json
```

출력: 평가셋×모델 평균 SDR 표 + **트랙별 페어드 비교**(평균 Δ, 향상 곡 수,
최대/최소 Δ). 트랙별 원자료는 JSON으로 저장되어 보고서 그림에 쓸 수 있다.

에폭별 추이를 보려면 여러 체크포인트를 한 번에 비교해도 된다:
`--model ep10=... --model ep30=... --model ep50=...` (표가 모델 수만큼 늘어난다).

## 5. 판정 기준

| 결과 | 해석 | 조치 |
|---|---|---|
| A에서 Δ ≥ +0.3 dB, B/C에서 Δ ≥ −0.1 dB | ✅ 성공 — 장르 특화 효과 확인 | 채택, 보고서 기록 |
| A 향상, B/C에서 −0.1 ~ −0.5 dB | ⚠️ 경미한 망각 | 용도 분리(장르 전용 모델로 명시) 또는 lr↓/에폭↓ 재시도 |
| A 향상, B/C에서 < −0.5 dB | ❌ catastrophic forgetting | lr 5e-6로 감소, 에폭 축소, 학습 데이터에 타 장르 소량 혼합, LoRA 전환 검토 |
| A에서 Δ < +0.1 dB | 효과 없음 | 데이터 양(40곡+)·품질(check_dataset) 점검, 에폭/lr 조정 |

Δ 기준은 커뮤니티 파인튜닝 선례(41곡에서 ~0.1-0.3 dB 변화가 유의미)에 맞춘
출발점이며, 트랙별 분산이 크면(±2 dB) 평균과 함께 **향상 곡 비율**로 판단한다.

## 6. 보고서 기록 양식

| 실험 | 학습 데이터 | 에폭/lr | A: pop (Δ) | B: others (Δ) | C: MUSDB test (Δ) | 판정 |
|---|---|---|---|---|---|---|
| pop-ft-01 | MoisesDB pop N곡 | 30 / 1e-5 | +0.4 | −0.05 | −0.02 | ✅ |

청취 평가(정성)도 병기 권장: A/B 각 2곡씩 base vs ft 블라인드로 듣고
보컬 잔향(bleeding)·아티팩트 유무 기록.
