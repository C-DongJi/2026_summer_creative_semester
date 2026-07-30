# MoisesDB 장르 선택 재학습 절차

MoisesDB(240곡, 12개 장르 메타데이터)를 원하는 장르만 골라 2-stem 학습셋으로
변환하고 Kim 체크포인트를 파인튜닝하는 전체 절차. GPU가 있는 환경(Colab 또는
5060 Ti PC — [SETUP_GPU_PC.md](SETUP_GPU_PC.md)) 기준이며, 1~3단계는 CPU로도 가능하다.

> 라이선스: MoisesDB는 CC BY-NC-SA 4.0(비상업 연구용). 이 데이터로 만든 파인튜닝
> 가중치는 상업적 이용이 제한될 수 있다 ([DATASETS.md](DATASETS.md) 참고).

## 0. 데이터 다운로드 (1회, 사람이 직접)

1. https://music.ai/research/ 접속 → datasets → MoisesDB → 신청 폼(이메일) 제출
2. 받은 링크로 다운로드 (약 80GB 단일 아카이브 — 부분 다운로드는 제공되지 않음.
   유선 연결로 밤 사이에 받는 것을 권장)

### 80GB 대응: 통째로 풀지 말고 장르만 선별 추출

전체 압축 해제는 디스크 160GB가 필요하다. 대신 zip 안의 메타데이터만 읽어
원하는 장르 트랙만 꺼내는 스크립트를 쓰면 zip(80GB) + 추출분(수 GB)로 끝난다:

```bash
# 장르별 트랙 수 미리 보기 (추출 없음)
python scripts/extract_moisesdb_subset.py --zip moisesdb.zip --list-genres

# 팝만 추출 (학습용) / 타 장르 추출 (평가셋 B용)
python scripts/extract_moisesdb_subset.py --zip moisesdb.zip --genres pop --out /data/moisesdb_pop
python scripts/extract_moisesdb_subset.py --zip moisesdb.zip --genres rock jazz electronic --out /data/moisesdb_others
```

변환(2단계)까지 끝나면 zip은 외장하드로 옮기거나 삭제해도 된다
(최종 점유: 변환된 2-stem 데이터 ~수 GB).

3. 압축 해제(또는 추출) 경로를 config에 기록해 두면 이후 명령이 짧아진다:
   ```yaml
   # config/default.yaml
   public_datasets:
     moisesdb_dir: /data/moisesdb_pop/moisesdb_v0.1   # 추출 경로 (스크립트 안내 참고)
   ```

## 1. 패키지 설치 (1회)

```bash
conda activate changuihakgi
pip install git+https://github.com/moises-ai/moises-db.git
```

## 2. 장르 선택 변환

MoisesDB의 장르 메타데이터(pop, rock, singer-songwriter, hip-hop/rap, electronic,
jazz 등 12종)를 부분 일치로 필터링한다. 보컬 없는 트랙은 자동으로 건너뛴다.

```bash
# 팝 장르만 -> 학습셋 + 검증셋(10%) 분리
python scripts/prepare_dataset.py --dataset moisesdb --genres pop

# 여러 장르 조합도 가능
python scripts/prepare_dataset.py --dataset moisesdb --genres pop rock

# 경로를 config에 안 적었다면 --src로 직접 지정
python scripts/prepare_dataset.py --dataset moisesdb --src /data/moisesdb_v0.1 --genres pop
```

결과: `data/processed/moisesdb__<트랙>/{vocals.wav, other.wav}` + `data/valid/`에
검증용 트랙이 결정적으로 분리된다.

## 3. 학습셋 점검 (필수)

```bash
python scripts/check_dataset.py --dir data/processed
python scripts/check_dataset.py --dir data/valid
```

이상 트랙(무음 보컬, 길이 불일치, 클리핑 등)이 보고되면 해당 폴더를 지우고 진행.
곡 수가 ~40 미만이면 장르를 넓히거나 다른 데이터를 보충한다.

## 4. 파인튜닝 실행 (GPU)

### 경로 A — MSST train.py (검증된 기본 경로)

```bash
python third_party/Music-Source-Separation-Training/train.py \
  --model_type mel_band_roformer \
  --config_path config/msst_finetune.yaml \
  --start_check_point models/checkpoints/MelBandRoformer.ckpt \
  --results_path models/checkpoints/finetune_pop \
  --data_path data/processed --valid_path data/valid \
  --dataset_type 4 \
  --num_workers 4 --pin_memory \
  --metrics sdr --metric_for_scheduler sdr
```

- VRAM 프리셋은 [config/msst_finetune.yaml](../config/msst_finetune.yaml)에 이미 반영
  (16GB면 `audio.chunk_size`를 352800으로 올려도 됨)
- ⚠️ 데이터를 바꾼 뒤에는 `metadata_*.pkl` 캐시를 삭제할 것 (변경이 조용히 무시됨)
- 홀드아웃 SDR이 정체되면 중단 — 소규모 데이터는 수십 에폭 내 수렴

### 경로 B — 자체 학습 루프 (창의학기제 학습 목표)

```bash
python scripts/train.py --config config/default.yaml
```
(`data.processed_dir`의 데이터로 학습, `training.resume_from`의 체크포인트에서 시작)

### 경로 C — Web UI (소규모, 사용자별)

Web UI의 "재학습" 탭에서 (mix, inst) 쌍을 직접 업로드해 학습할 수도 있다 —
MoisesDB처럼 스템이 이미 분리된 데이터는 경로 A/B가 적합하고, C는 사용자가
개별 수집한 곡 소량을 얹을 때 쓴다.

## 5. 결과 사용

- 파인튜닝 체크포인트를 `models/checkpoints/`(또는 `users/<이름>/`) 아래에 두면
  **Web UI의 모델 드롭다운에 자동으로 나타난다** → 선택해서 분리 품질 비교
- CLI 비교: `config/default.yaml`의 `model.checkpoint`를 새 경로로 바꾸고
  `python scripts/separate.py --input <곡>` 실행
- 평가: **재학습 장르 + 타 장르를 함께** SDR로 비교해 향상과 부작용(망각)을 동시에
  확인한다 — 실험 설계·판정 기준·명령어는 [EVALUATION.md](EVALUATION.md), 실행은
  `scripts/evaluate.py`
