"""이 파일은 대시보드 로더가 streamlit 없이도 순수 함수로 동작하는지 확인한다.
입력: 임시 디렉터리의 가짜 산출물.
출력: pytest 통과/실패.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import loaders  # noqa: E402


def test_산출물_목록은_npz만_돌려준다(tmp_path):
    (tmp_path / "a.npz").write_bytes(b"x")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "c.npz").write_bytes(b"y")
    결과 = loaders.list_runs(tmp_path)
    assert [p.name for p in 결과] == ["a.npz", "c.npz"]


def test_산출물_폴더가_없으면_빈_목록(tmp_path):
    assert loaders.list_runs(tmp_path / "없음") == []


def test_비교보고서를_읽는다(tmp_path):
    경로 = tmp_path / "supervised_comparison.json"
    경로.write_text(json.dumps({"schema_version": 1, "dp_ev": -0.005108}),
                    encoding="utf-8")
    실린값 = loaders.read_comparison(경로)
    assert 실린값["schema_version"] == 1


def test_비교보고서가_없으면_None(tmp_path):
    assert loaders.read_comparison(tmp_path / "없음.json") is None


def test_스키마_버전이_다르면_예외(tmp_path):
    # 왜: 포맷이 바뀐 옛 보고서를 조용히 그리면 화면의 숫자가 틀리게 된다.
    경로 = tmp_path / "old.json"
    경로.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError):
        loaders.read_comparison(경로)


def test_로더는_streamlit_없이_임포트된다():
    # 왜: 이 테스트가 app/ 과 src/ 의 경계를 지킨다. 로더의 순수 함수들이
    #     streamlit에 묶여 있으면 테스트할 수 없다.
    import subprocess

    코드 = (
        "import sys; sys.path.insert(0, 'app'); "
        "sys.modules['streamlit'] = None; "
        "import loaders; print(loaders.list_runs('없는폴더'))"
    )
    결과 = subprocess.run([sys.executable, "-c", 코드], capture_output=True, text=True)
    assert 결과.returncode == 0, 결과.stderr
