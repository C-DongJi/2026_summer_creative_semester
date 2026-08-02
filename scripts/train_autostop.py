"""MSST 학습을 valid 지표 정체 시 자동 중단하는 래퍼.

MSST train.py에는 조기 종료가 없어 valid SDR이 정체돼도 num_epochs까지 계속
돈다. 이 래퍼는 학습 명령을 자식 프로세스로 실행하면서 출력을 그대로 흘려보내고,
'Train epoch: N'(에폭 시작)과 'Store weights: ...'(베스트 갱신 시에만 출력)
라인을 감시해 베스트 갱신 없이 --patience 에폭이 지나면 Ctrl+C와 같은 방식
(SIGINT)으로 학습을 멈춘다. 베스트 체크포인트는 갱신 시점마다 이미 저장돼
있으므로 잃는 것이 없다.

사용 예 (기존 학습 명령 앞에 래퍼만 붙이면 됨):
    python scripts/train_autostop.py --patience 5 --log outputs/train_pop.log -- \
      python third_party/Music-Source-Separation-Training/train.py \
        --model_type mel_band_roformer ... (이하 기존 명령 그대로)

--log 파일에는 stdout만 저장된다 (tqdm 진행바는 stderr라 화면에만 나오고
로그는 깔끔하게 남는다). 자동 중단 시 마지막 베스트 체크포인트 경로를 출력한다.
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

EPOCH_RE = re.compile(r"^Train epoch:\s*(\d+)\b")
BEST_RE = re.compile(r"^Store weights:\s*(.+)")


def _terminate(proc: subprocess.Popen) -> None:
    """자식 프로세스 그룹(dataloader worker 포함)을 SIGINT→SIGKILL 순으로 정리."""
    for sig, timeout in ((signal.SIGINT, 30), (signal.SIGTERM, 15)):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="valid 정체 시 MSST 학습 자동 중단 (-- 뒤에 학습 명령)")
    parser.add_argument("--patience", type=int, default=5,
                        help="베스트 갱신 없이 허용할 에폭 수 (기본 5)")
    parser.add_argument("--log", default=None,
                        help="stdout을 저장할 로그 파일 (tee 대체)")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="-- 뒤에 원래 학습 명령 전체")
    args = parser.parse_args()

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        parser.error("-- 뒤에 학습 명령을 넣으세요. (예: -- python .../train.py ...)")

    log_f = None
    if args.log:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log, "w", encoding="utf-8")

    def emit(line: str) -> None:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log_f:
            log_f.write(line)
            log_f.flush()

    # 자식 python이 파이프에 블록 버퍼링하면 감시가 늦어지므로 무버퍼 강제.
    # 새 세션으로 띄워 SIGINT를 학습 프로세스 그룹에만 정확히 보낸다.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    epoch: int | None = None
    best_epoch: int | None = None
    best_path: str | None = None
    auto_stopped = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            emit(line)
            m = EPOCH_RE.match(line)
            if m:
                epoch = int(m.group(1))
                if best_epoch is None:
                    best_epoch = epoch  # 첫 valid 전 기준점
                waited = epoch - best_epoch
                emit(f"[autostop] epoch {epoch} 시작 — 마지막 베스트: epoch "
                     f"{best_epoch} (정체 {waited}/{args.patience})\n")
                if waited > args.patience:
                    emit(f"[autostop] {args.patience}에폭 연속 베스트 갱신 없음 "
                         f"→ 학습 자동 중단\n")
                    auto_stopped = True
                    _terminate(proc)
                    break
                continue
            m = BEST_RE.match(line)
            if m and epoch is not None:
                best_epoch = epoch
                best_path = m.group(1).strip()
    except KeyboardInterrupt:
        emit("\n[autostop] 수동 중단(Ctrl+C) — 학습 프로세스 정리 중\n")
        _terminate(proc)
    finally:
        ret = proc.wait()
        if best_path:
            emit(f"[autostop] 최종 베스트 체크포인트: {best_path}\n")
        else:
            emit("[autostop] 저장된 베스트 없음 (첫 valid 전에 종료됨)\n")
        if log_f:
            log_f.close()
        # 자동/수동 중단은 정상 종료로 취급
        sys.exit(0 if auto_stopped else ret)


if __name__ == "__main__":
    main()
