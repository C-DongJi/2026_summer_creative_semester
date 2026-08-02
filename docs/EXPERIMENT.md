# 재학습 실험 절차서 — 데이터 준비부터 결과 도출까지

5060 Ti PC에서 "장르 파인튜닝이 효과가 있는가"를 실험하고 **보고서에 쓸 결과
표까지 뽑는** 전체 절차. 팝(pop) 장르를 예시로 쓰며, 다른 장르는 `--genres`만
바꾸면 된다. 설계 근거와 판정 기준의 상세는 [EVALUATION.md](EVALUATION.md) 참고.

> 전제: [SETUP_GPU_PC.md](SETUP_GPU_PC.md) 설치와 [TESTING.md](TESTING.md)의
> 기능 테스트(1~4단계)가 끝난 상태. 소요 시간: 데이터 준비 ~1시간(다운로드 제외),
> 학습 수 시간, 평가 ~30분.

---

## 0. 데이터 확보 (1회, 사람이 직접)

https://music.ai/research/ 에서 MoisesDB 신청·다운로드. 80GB zip이므로 통째로 풀지
말고 필요한 장르만 선별 추출한다 — 추출 방법과 Windows D드라이브 연결(`/mnt/d`)은
[FINETUNE_MOISESDB.md](FINETUNE_MOISESDB.md) 0단계 참고.
아래에서 추출(압축 해제) 경로를 `<MOISESDB>`로 표기한다.

```bash
pip install git+https://github.com/moises-ai/moises-db.git   # 1회
```

## 1. 실험 데이터셋 구성 (~30분)

세 덩어리를 만든다: 학습셋(팝), 평가셋 A(팝 홀드아웃), 평가셋 B(타 장르).

```bash
# 학습셋 + 평가셋 A: 팝만 변환, 10%는 자동으로 평가용 분리
python scripts/prepare_dataset.py --dataset moisesdb --src <MOISESDB> \
    --genres pop --out data/train_pop --valid-dir data/eval/pop

# 평가셋 B: 타 장르 (학습에 안 씀 — 전부 평가용)
python scripts/prepare_dataset.py --dataset moisesdb --src <MOISESDB> \
    --genres rock jazz electronic --out data/eval/others --valid-dir data/eval/others --holdout-frac 0
```

점검 3종 (모두 통과해야 다음 단계로):

```bash
# (1) 데이터 품질
python scripts/check_dataset.py --dir data/train_pop
python scripts/check_dataset.py --dir data/eval/pop
python scripts/check_dataset.py --dir data/eval/others

# (2) 누설(leakage): 출력이 비어 있어야 함
comm -12 <(ls data/train_pop | sort) <(ls data/eval/pop data/eval/others | sort)

# (3) 규모 확인: 학습 30곡+, 평가 A 5곡+, B 10곡+ 권장
ls data/train_pop | wc -l; ls data/eval/pop | wc -l; ls data/eval/others | wc -l
```

학습셋이 30곡 미만이면 `--genres pop singer-songwriter`처럼 인접 장르를 합친다.

## 2. 파인튜닝 실행 (수 시간, 자동 중단 포함)

MSST train.py에는 조기 종료가 없어 정체돼도 계속 돌기 때문에, 감시 래퍼를
붙여 실행한다 — valid SDR이 `--patience` 에폭 연속 갱신되지 않으면 자동으로
멈추고, 로그 저장과 종료 시 베스트 체크포인트 경로 출력까지 해준다:

```bash
python scripts/train_autostop.py --patience 5 --log outputs/train_pop.log -- \
  python third_party/Music-Source-Separation-Training/train.py \
  --model_type mel_band_roformer \
  --config_path config/msst_finetune.yaml \
  --start_check_point models/checkpoints/MelBandRoformer.ckpt \
  --results_path models/checkpoints/finetune_pop \
  --data_path data/train_pop --valid_path data/eval/pop \
  --dataset_type 4 --num_workers 4 --pin_memory \
  --metrics sdr --metric_for_scheduler sdr
```

- 16GB 카드면 시작 전에 `config/msst_finetune.yaml`의 `audio.chunk_size`를
  352800으로 올려도 된다 (OOM 나면 131584로 복귀)
- 베스트 체크포인트는 갱신 시점마다 저장되므로 자동 중단으로 잃는 것은 없다.
  소규모 데이터는 보통 수십 에폭 내 수렴
- 래퍼 없이 돌리려면 `--` 뒤의 명령만 실행하고 valid SDR을 직접 지켜보다
  정체 시 Ctrl+C (로그는 명령 끝에 `2>&1 | tee outputs/train_pop.log`)
- 데이터를 바꿔 재실행할 땐 `metadata_*.pkl` 캐시 삭제 필수

완료되면 `models/checkpoints/finetune_pop/` 안의 **valid SDR이 가장 높았던
체크포인트**를 고른다 (파일명에 SDR이 붙는다). 아래에서 `<BEST>`로 표기.

## 3. 평가 실행 (~30분)

base와 파인튜닝 모델을 같은 평가셋 A/B에서 한 번에 비교한다:

```bash
python scripts/evaluate.py \
  --eval-set pop=data/eval/pop \
  --eval-set others=data/eval/others \
  --model base=models/checkpoints/MelBandRoformer.ckpt \
  --model pop_ft=models/checkpoints/finetune_pop/<BEST> \
  --output outputs/eval_pop_ft.json
```

출력 두 가지를 그대로 기록한다:
- `=== 평균 SDR (dB) ===` 표: 평가셋 × 모델의 vocals/inst 평균
- `=== 페어드 비교: pop_ft − base ===`: 평가셋별 평균 Δ, 향상 곡 수, 최대/최소 Δ

## 4. 결과 해석 및 판정

[EVALUATION.md](EVALUATION.md) 5절 기준표 적용:

| pop (A) 평균 Δ | others (B) 평균 Δ | 판정 |
|---|---|---|
| ≥ +0.3 dB | ≥ −0.1 dB | ✅ 장르 특화 성공 |
| ≥ +0.3 dB | −0.1 ~ −0.5 dB | ⚠️ 경미한 망각 — 장르 전용 모델로 명시하거나 lr↓ 재시도 |
| ≥ +0.3 dB | < −0.5 dB | ❌ 망각 — lr 5e-6, 에폭 축소, 타 장르 혼합, LoRA 검토 |
| < +0.1 dB | — | 효과 없음 — 데이터 양/품질 점검, 하이퍼파라미터 조정 |

평균이 애매하면(트랙별 분산 ±2dB) **향상 곡 비율**(페어드 비교의 "향상 N/M곡")로
판단한다.

## 5. 보고서 생성 (자동)

평가 JSON에서 보고서 파일을 자동으로 만든다 — 요약 표, 페어드 비교,
기준표 자동 판정, 청취 기록란, 트랙별 부록 포함:

```bash
python scripts/make_report.py --results outputs/eval_pop_ft.json \
    --target-set pop --experiment-name pop-ft-01 \
    --train-data "MoisesDB pop N곡" --epochs-lr "30 / 1e-5"
# -> outputs/report_pop-ft-01.md
```

생성된 .md는 그대로 제출하거나 최종 보고서(한글/워드)에 붙여넣는다.
청취 평가 표(4절)만 수동으로 채우면 된다.

수동 기록용 요약 행 형식 (자동 보고서에도 동일 수치가 있음):

```
| 실험 | 학습 데이터 | 에폭/lr | A: pop Δ | B: others Δ | 향상률(A) | 판정 |
|------|-----------|---------|----------|-------------|----------|------|
| pop-ft-01 | MoisesDB pop N곡 | E / 1e-5 | +0.xx dB | -0.xx dB | n/m | ✅/⚠️/❌ |
```

- 근거 자료 보존: `outputs/eval_pop_ft.json`(트랙별 수치), `outputs/train_pop.log`
  (학습 곡선), 2단계에서 고른 체크포인트 경로
- 정성 평가 병기: A/B 각 2곡을 base vs ft로 분리해 블라인드 청취,
  보컬 잔향·아티팩트 차이 메모
- 웹 데모 연결: `<BEST>`가 `models/checkpoints/` 아래에 있으므로 Web UI
  드롭다운에서 바로 시연 가능

## 6. (선택) 추가 실험

- **에폭별 추이**: `--model ep10=... --model ep30=... --model best=...`로
  한 번에 비교해 과적합 시점 파악
- **판정이 ⚠️/❌일 때 재시도 우선순위**: lr 1e-5 → 5e-6 → 에폭 절반 →
  학습셋에 타 장르 20% 혼합 → LoRA(`--train_lora_peft`, 사용 전 config에
  `lora:` 섹션 추가 필요 — msst_finetune.yaml 상단 주석 참고)
- **외부 표준셋(C)**: MUSDB18-HQ test 50곡으로 같은 평가를 반복하면 공개
  리더보드와 비교 가능한 수치를 얻는다 (변환 스크립트 필요 시 요청)
