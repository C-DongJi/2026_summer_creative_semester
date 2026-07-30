# 5060 Ti PC 구동 매뉴얼

RTX 5060 Ti가 있는 PC에서 이 프로젝트 전체(분리 데모, GPU 검증, 파인튜닝, Web UI)를
돌리기 위한 처음부터 끝까지의 절차. 위에서부터 순서대로 따라 하면 된다.

5060 Ti는 8GB와 16GB 두 버전이 있다. 자기 버전을 모르면 2단계의 `nvidia-smi`
출력(memory.total)으로 확인하고, 6단계 프리셋 표에서 해당 열을 따르면 된다.

---

## 1. 꼭 알아야 할 것: PyTorch는 CUDA 12.8 빌드

RTX 50 시리즈(Blackwell, sm_120)는 **CUDA 12.8 이상 빌드의 PyTorch가 필수**다.
다른 문서에 나오는 cu121 명령을 쓰면 `sm_120 is not compatible` 오류가 나며 GPU를
전혀 쓰지 못한다. 이 매뉴얼의 설치 명령(cu128)만 따르면 문제없다.

## 2. Windows 사전 준비 (약 20분)

1. **NVIDIA 드라이버**: https://www.nvidia.com/drivers 에서 최신 버전 설치
   (RTX 50 지원 버전). 설치 후 PowerShell에서 `nvidia-smi` 실행, 5060 Ti가 보여야 함.
2. **WSL2 + Ubuntu** (관리자 PowerShell):
   ```powershell
   wsl --install -d Ubuntu
   ```
   재부팅 후 Ubuntu 초기 계정을 만든다. Windows 드라이버만 있으면 WSL 안에서
   GPU가 자동 인식된다(WSL용 CUDA 툴킷 별도 설치 불필요).
   WSL 터미널에서 `nvidia-smi`가 나오는지 확인.

   이어서 빌드 도구를 설치한다 (일부 파이썬 패키지가 C 컴파일러를 요구):
   ```bash
   sudo apt update && sudo apt install -y build-essential
   ```
3. **Miniconda** (WSL 터미널):
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh    # 기본값으로 진행, 마지막 init yes
   ```
   설치 후 터미널을 한 번 닫았다 다시 연다.

## 3. 프로젝트 설치 (약 15분 + 다운로드)

```bash
# 코드
git clone -b leejy https://github.com/C-DongJi/2026_summer_creative_semester.git
cd 2026_summer_creative_semester

# 환경 (Python 3.11 + ffmpeg)
conda create -y -n changuihakgi python=3.11
conda activate changuihakgi
conda install -y -c conda-forge ffmpeg

# ★ 5060 Ti용 CUDA 빌드 torch (반드시 cu128)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 나머지 의존성
pip install soundfile librosa einops tqdm pyyaml gradio pytest

# MSST 프레임워크 + Kim 체크포인트(약 913MB) 자동 설치
bash scripts/setup_msst.sh
```

다운로드가 중간에 끊겨도 `bash scripts/setup_msst.sh`를 다시 실행하면 이어받는다.

## 4. 동작 확인 체크리스트 (약 5분)

순서대로 실행해서 전부 통과해야 다음으로 넘어간다.

```bash
# (1) GPU 인식: True + "NVIDIA GeForce RTX 5060 Ti"가 나와야 함
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# (2) 테스트 스위트: 전부 통과해야 함
python -m pytest tests/ -q

# (3) 오디오 I/O: "통과 8 / 8"
python scripts/verify_audio_io.py
```

(1)에서 False가 나오거나 sm_120 오류가 보이면 9단계 문제 해결 참고.

## 5. GPU 검증 실행 (2·3주차 실측, 약 20분)

Colab 없이 이 PC에서 바로 하면 된다. 결과는 [SCHEDULE.md](SCHEDULE.md)의
GPU 세션 ① 체크리스트와 보고서에 기록한다.

```bash
# 길이를 늘려가며 베이스라인(청크 없음) vs 청크 추론의 피크 VRAM 비교
python scripts/verify_oom_chunking.py --minutes 1
python scripts/verify_oom_chunking.py --minutes 2
python scripts/verify_oom_chunking.py --minutes 4
python scripts/verify_oom_chunking.py --minutes 8
```

기대 결과: 베이스라인은 길이에 따라 VRAM이 급증하다 OOM, 청크 방식은 일정 VRAM으로
전부 성공. 각 실행의 상태·시간·피크 VRAM을 표로 남길 것.

실제 곡으로도 확인:
```bash
# 아무 곡이나 (mp3/wav). 분리 결과가 outputs/에 저장된다
python scripts/separate.py --input <곡파일> --output outputs/
```
들어보고 확인할 것: 보컬/반주가 분리되는지, 청크 경계(2초 간격으로 지나감)에서
틱 소리가 없는지(Overlap-Add 검증 포인트).

## 6. VRAM 프리셋

| 항목 | 5060 Ti 8GB | 5060 Ti 16GB |
|---|---|---|
| 추론 | 기본 설정 그대로 | 기본 + `inference.batch_size: 2~4` |
| 학습 chunk_size (`config/msst_finetune.yaml`) | 131584 (기본값 유지) | 352800으로 상향 가능 |
| 학습 옵션 | `optimizer: adamw8bit` + grad accum 8 | adam + grad accum 4~8 |

추론은 `inference.precision: auto`가 fp16을 자동 적용해 VRAM을 절반으로 쓴다.
학습 중 OOM이 나면: chunk_size를 한 단계 낮추기 → adamw8bit → LoRA 순으로 시도.
LoRA(`--train_lora_peft`)는 쓰기 전에 `config/msst_finetune.yaml`에 `lora:` 섹션
(r: 8, lora_alpha: 16 등)을 추가해야 한다 — 없으면 KeyError로 죽는다.

## 7. Web UI (분리 + 사용자별 재학습)

```bash
# 이 PC에서만 사용
python app/webui.py
# 브라우저에서 http://localhost:7860

# 같은 공유기의 다른 기기(노트북·폰)에서도 접속 허용
python app/webui.py --host 0.0.0.0
# 다른 기기 브라우저에서 http://<이 PC IP>:7860
# (Windows 방화벽이 물으면 허용. 안 열리면 7860 인바운드 규칙 추가)
```

- **음원 분리 탭**: 곡 업로드 → 분리 → 보컬/반주 재생·다운로드. 모델 드롭다운에서
  기본 모델과 파인튜닝 모델을 골라 비교할 수 있다. GPU라서 곡당 수 초~수십 초.
- **재학습 탭**: `<곡명>_mix.wav` / `<곡명>_inst.wav` 규칙의 쌍 파일을 올리고
  사용자 이름을 입력하면 사용자별 데이터셋이 쌓인다. 재학습 시작을 누르면 이
  PC의 GPU로 학습되고, 끝나면 분리 탭 드롭다운에 "<이름>님의 모델"이 나타난다.

## 8. 파인튜닝 (장르 특화)

MoisesDB에서 원하는 장르만 골라 재학습하는 전체 절차는
[FINETUNE_MOISESDB.md](FINETUNE_MOISESDB.md)에 있다. 요약:

```bash
# 데이터: music.ai/research 폼 신청 후 다운로드·압축해제 (사람이 1회)
pip install git+https://github.com/moises-ai/moises-db.git

# 팝 장르만 2-stem 학습셋으로 변환 + 점검
python scripts/prepare_dataset.py --dataset moisesdb --src <압축해제경로> --genres pop
python scripts/check_dataset.py --dir data/processed

# 파인튜닝 (MSST 경로 — 상세 옵션은 config/msst_finetune.yaml 상단 주석)
python third_party/Music-Source-Separation-Training/train.py \
  --model_type mel_band_roformer \
  --config_path config/msst_finetune.yaml \
  --start_check_point models/checkpoints/MelBandRoformer.ckpt \
  --results_path models/checkpoints/finetune_pop \
  --data_path data/processed --valid_path data/valid \
  --dataset_type 4 --num_workers 4 --pin_memory \
  --metrics sdr --metric_for_scheduler sdr
```

완료되면 `models/checkpoints/finetune_pop/` 안의 MSST 결과물(`model_*_sdr_*.ckpt`)이
Web UI 드롭다운에 자동으로 나타난다 (자체 루프 결과 `finetune_epoch*.ckpt`도 동일).

## 9. 문제 해결

| 증상 | 조치 |
|---|---|
| `error: command 'gcc' failed` (pesq/diffq 빌드 실패) | 빌드 도구 미설치. `sudo apt update && sudo apt install -y build-essential` 후 `bash scripts/setup_msst.sh` 재실행 |
| `sm_120 is not compatible ...` | torch가 cu128이 아님. `pip uninstall torch torchaudio` 후 3단계 cu128 명령으로 재설치 |
| `No available kernel. Aborting execution.` | flash attention은 반정밀 필요. `git pull` 후 재실행(자동 적용) |
| `view_as_complex is only supported for half, float and double ... BFloat16` | 이 모델은 bf16 불가(복소 변환 미지원). `git pull` 후 재실행하면 auto가 fp16을 적용. 수동 지정은 verify_oom_chunking.py는 `--precision fp16`, separate.py/Web UI는 config의 `inference.precision: fp16` |
| WSL에서 `nvidia-smi` 없음 | Windows 드라이버 구버전이거나 WSL 재시작 필요: PowerShell에서 `wsl --shutdown` 후 재진입 |
| `torch.cuda.is_available()` False | 위 두 항목 순서로 확인 |
| 체크포인트 다운로드 중단 | `bash scripts/setup_msst.sh` 재실행(이어받기). "다운로드 불완전" 메시지가 나오면 한 번 더 |
| 학습 OOM | 6단계 프리셋 표의 낮은 사양 열 적용 (chunk_size ↓ → adamw8bit → LoRA) |
| MSST 학습이 데이터 변경을 무시 | `metadata_*.pkl` 캐시 삭제 후 재실행 |
| Web UI가 다른 기기에서 안 열림 | `--host 0.0.0.0`으로 실행했는지, Windows 방화벽 7860 인바운드 허용했는지 확인 |

## 10. 협업 규칙 (git)

- 작업은 각자 이름 브랜치에서 (이준영: `leejy`). main 직접 커밋은 하지 않는다.
- 커밋 메시지는 영어 Conventional Commits (`type(scope): imperative ...`).
- 대용량(체크포인트·데이터·third_party)은 .gitignore에 이미 제외되어 있으니
  실수로 `git add -f` 하지 않도록 주의.
