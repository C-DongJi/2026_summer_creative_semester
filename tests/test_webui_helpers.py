"""Web UI 순수 헬퍼 테스트 (서버 실행 없이)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.webui import pair_uploads, sanitize_user


def test_sanitize_user():
    assert sanitize_user("  leejy ") == "leejy"
    assert sanitize_user("이준영_01") == "이준영_01"
    assert sanitize_user("../..//etc") == "etc"          # 경로 탈출 문자는 제거
    assert sanitize_user("a b!c@d") == "abcd"
    assert sanitize_user(None) == ""
    assert len(sanitize_user("x" * 100)) == 40


def test_pair_uploads_basic():
    files = ["/tmp/songA_mix.wav", "/tmp/songA_inst.wav",
             "/tmp/songB_mix.mp3", "/tmp/songB_inst.mp3"]
    pairs, incomplete = pair_uploads(files)
    assert set(pairs) == {"songA", "songB"}
    assert incomplete == []
    mix, inst = pairs["songA"]
    assert mix.name == "songA_mix.wav" and inst.name == "songA_inst.wav"


def test_pair_uploads_incomplete_and_case():
    files = ["/tmp/songA_MIX.wav",       # 대문자 접미사 허용
             "/tmp/songA_inst.wav",
             "/tmp/lonely_mix.wav",      # inst 없음
             "/tmp/readme.txt"]          # 규칙 미준수 -> 무시
    pairs, incomplete = pair_uploads(files)
    assert set(pairs) == {"songA"}
    assert incomplete == ["lonely"]


def test_pair_uploads_empty():
    pairs, incomplete = pair_uploads([])
    assert pairs == {} and incomplete == []
    pairs, incomplete = pair_uploads(None)
    assert pairs == {} and incomplete == []


def test_list_checkpoints_discovers_all_layouts(tmp_path, monkeypatch):
    """기본/자체 루프/MSST 결과물/사용자별 체크포인트가 모두 목록에 떠야 한다."""
    import app.webui as webui

    (tmp_path / "MelBandRoformer.ckpt").touch()
    (tmp_path / "finetune_epoch3.ckpt").touch()
    msst_dir = tmp_path / "finetune_pop"
    msst_dir.mkdir()
    (msst_dir / "model_mel_band_roformer_ep_12_sdr_11.0341.ckpt").touch()
    user_dir = tmp_path / "users" / "leejy"
    user_dir.mkdir(parents=True)
    (user_dir / "finetune_epoch5.ckpt").touch()

    monkeypatch.setattr(webui, "CKPT_DIR", tmp_path)
    paths = [p for _, p in webui.list_checkpoints()]

    assert str(tmp_path / "MelBandRoformer.ckpt") in paths
    assert str(tmp_path / "finetune_epoch3.ckpt") in paths
    assert str(msst_dir / "model_mel_band_roformer_ep_12_sdr_11.0341.ckpt") in paths
    assert str(user_dir / "finetune_epoch5.ckpt") in paths
    assert len(paths) == 4
