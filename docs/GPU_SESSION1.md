# GPU 세션 ① Runbook — 베이스라인 OOM 테스트 + 청크 로직 실증

> 2주차(베이스라인 구동·OOM 관찰) + 3주차 검증(청크 분할 추론 실증)을 Colab에서 한 번에 진행.
> 아래 명령을 Colab 터미널(VS Code 터널 또는 Colab 터미널)에 순서대로 붙여넣으면 된다.
> 터널 연결 방법: [COLAB.md](COLAB.md)

## 0. 필요한 리소스: **T4 (무료 티어)로 충분**

| 항목 | 추정 | 근거 |
|---|---|---|
| 모델 가중치 (Kim, fp32) | ~1.0 GB | ~2.4억 파라미터 × 4B |
| 청크 추론(8초) 활성값 | +3~5 GB | 8초 = STFT 약 800프레임, 시간축 어텐션 800² |
| **청크 방식 합계** | **~6 GB** | → T4(16GB)에 여유 있게 수용 |
| 베이스라인(4분 일괄) | 수십 GB 필요 | 240초 = 약 24,000프레임 → 어텐션 메모리가 프레임 수의 **제곱**(24,000² ≈ 800²의 900배) |

핵심: RoFormer의 시간축 어텐션 메모리는 입력 길이의 **제곱**으로 늘어난다.
- **T4 16GB**: 베이스라인은 3~4분 곡에서 OOM 예상, 청크 방식은 통과 → **우리가 보여주려는 대비가 정확히 재현됨.** 오히려 작은 GPU가 시연에 유리하다.
- **A100 불필요**: A100(40GB)은 베이스라인이 더 긴 길이까지 버텨서 OOM 시연에 불리하고, 청크 검증에는 과사양. 파인튜닝(GPU 세션 ②)도 커뮤니티 선례상 8GB에서 가능하므로 T4/L4면 충분하다.
- 만약 T4에서 4분 베이스라인이 OOM 없이 통과하면 `--minutes 8, 12...`로 늘려 한계 지점을 기록하면 된다.

## 1. 환경 준비 (~10분, 체크포인트 다운로드 포함)

```bash
# 코드 (leejy 브랜치)
git clone -b leejy https://github.com/C-DongJi/2026_summer_creative_semester.git
cd 2026_summer_creative_semester

# 의존성 — ⚠️ torch/torchaudio는 재설치 금지 (Colab의 CUDA 빌드 유지)
pip install soundfile librosa einops tqdm pyyaml

# MSST 프레임워크 + Kim 체크포인트(~913MB)
bash scripts/setup_msst.sh

# GPU 확인
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 2. 오디오 I/O 검증 (1분) — 로컬에서 이미 통과, Colab 환경 재확인

```bash
python scripts/verify_audio_io.py
```
기대: `통과 8 / 8` (mp3/ogg가 스킵되면 libsndfile 구버전 — 기록만 하고 진행).

## 3. 【2주차】 베이스라인 OOM 한계 테스트

길이를 늘려가며 베이스라인(청크 없음)이 어디서 OOM 나는지 기록:

```bash
python scripts/verify_oom_chunking.py --minutes 1
python scripts/verify_oom_chunking.py --minutes 2
python scripts/verify_oom_chunking.py --minutes 4
python scripts/verify_oom_chunking.py --minutes 8
```

각 실행이 [A] 베이스라인 / [B] 청크 방식의 상태·시간·피크 VRAM을 출력한다.
아래 표를 채워 결과 보고서에 사용:

| 길이 | 베이스라인 | 베이스라인 피크 VRAM | 청크 방식 | 청크 피크 VRAM |
|------|-----------|--------------------|----------|---------------|
| 1분  |           |                    |          |               |
| 2분  |           |                    |          |               |
| 4분  |           |                    |          |               |
| 8분  |           |                    |          |               |

기대 결과: 베이스라인 피크 VRAM이 길이에 따라 급증하다 OOM,
청크 방식은 길이와 무관하게 ~일정. → **"OOM은 청크 분할로 해결된다" 실증 완료.**

## 4. 【3주차 검증】 실제 곡으로 품질 확인

실제 3~4분 곡(mp3/wav)을 Colab에 올린 뒤 (VS Code 탐색기 드래그 또는 구글 드라이브):

```bash
# 청크 파이프라인으로 실주행 + VRAM 측정
python scripts/verify_oom_chunking.py --input <곡파일>

# 분리 결과물 생성 (vocals/instrumental wav)
python scripts/separate.py --input <곡파일> --output outputs/
```

`outputs/`의 `*_vocals.wav`, `*_instrumental.wav`를 다운로드해서 청취 확인:
- [ ] 보컬/반주가 실제로 분리되는가
- [ ] **청크 경계(8초 간격)에서 틱/뚝 소리가 없는가** ← Overlap-Add 크로스페이드 검증 포인트
- [ ] 반주에 보컬 잔향(bleeding)이 심하지 않은가

## 5. 결과 기록 & 세션 종료 전 백업

Colab 디스크는 세션 종료 시 초기화된다. 나가기 전에:

```bash
# 결과물을 구글 드라이브로 (드라이브 마운트 상태에서)
cp -r outputs/ /content/drive/MyDrive/creative_semester_results/

# 체크포인트 캐시 (다음 세션 재다운로드 절약)
cp models/checkpoints/MelBandRoformer.ckpt /content/drive/MyDrive/msst/
```

측정 표와 청취 결과는 [SCHEDULE.md](SCHEDULE.md)의 GPU 세션 ① 항목에 기록.

## 트러블슈팅

- **`ModuleNotFoundError: ml_collections`** → `pip install ml_collections` (MSST requirements가 설치 안 된 경우)
- **베이스라인이 OOM인데 스크립트가 죽음(잡히지 않음)** → CUDA OOM이 아닌 시스템 RAM 부족일 수 있음. `--minutes`를 줄여서 재시도
- **청크 방식도 OOM** → config의 `inference.chunk_seconds`를 8.0 → 4.0으로 낮춰 재시도 (그 자체가 유의미한 측정 결과)
- **다운로드가 매우 느림** → 드라이브에 캐시한 체크포인트 사용 (COLAB.md 부록 A)
