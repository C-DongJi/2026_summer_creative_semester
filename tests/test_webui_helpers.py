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
