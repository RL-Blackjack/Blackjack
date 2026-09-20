"""이 파일은 앱 조립 계층이 잘못된 입력과 모델 적재를 어떻게 다루는지 확인한다.
입력: 원문 JSON으로 보낸 HTTP 요청과 임시 모델 폴더.
출력: 422 본문·헬스체크 모델 수·기동 실패에 대한 pytest 결과.
"""

import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.config import get_settings  # noqa: E402
from game.db import create_schema, make_engine, make_session_factory  # noqa: E402
from game.db import reset_engine  # noqa: E402

제이슨 = {"content-type": "application/json"}
가입 = '{"email":"a@example.com","password":"password123","display_name":%s}'


def _전역비우기() -> None:
    """프로세스 전역 상태(요청 제한기·모델 저장소)를 비운다."""
    # 왜: 제한기와 store는 모듈 전역이라 앞 테스트 파일의 가입 기록과 올린 모델이
    #   남는다. 그러면 파일 실행 순서에 따라 429가 나거나 모델 수가 틀어진다.
    from game.deps import login_limiter, signup_limiter
    from game.serving import store
    login_limiter.reset()
    signup_limiter.reset()
    store.reset()


def _환경(tmp_path, monkeypatch) -> None:
    """임시 SQLite를 가리키게 하고 설정 캐시와 전역 상태를 비운다."""
    monkeypatch.setenv("BJ_DATABASE_URL", f"sqlite:///{tmp_path / 'app.db'}")
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    _전역비우기()


def _모델등록(models_dir: Path) -> int:
    """지금 설정이 가리키는 DB에 models_dir의 정책을 레지스트리로 넣는다."""
    from scripts.register_models import sync_registry
    엔진 = make_engine()
    create_schema(엔진)
    with make_session_factory(엔진)() as 세션:
        개수 = sync_registry(세션, models_dir)
    엔진.dispose()
    return 개수


@pytest.fixture
def 클라(tmp_path, monkeypatch):
    """모델을 하나도 등록하지 않은 서버에 붙은 테스트 클라이언트."""
    _환경(tmp_path, monkeypatch)
    from game.main import create_app
    with TestClient(create_app()) as c:
        yield c
    _전역비우기()
    reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def 모델클라(tmp_path, monkeypatch):
    """저장소의 모델을 전부 등록한 서버에 붙은 (클라이언트, 등록 수)."""
    _환경(tmp_path, monkeypatch)
    등록수 = _모델등록(ROOT / "models")
    from game.main import create_app
    with TestClient(create_app()) as c:
        yield c, 등록수
    _전역비우기()
    reset_engine()
    get_settings.cache_clear()


# ---------- RequestValidationError 처리기 ----------

def test_NaN이_든_요청은_422다(클라):
    # 왜: Python의 json.loads는 NaN을 받아들이고 pydantic은 422로 거부한다. 그런데
    #     기본 처리기가 그 NaN을 응답에 되싣고 starlette는 allow_nan=False로
    #     직렬화해 ValueError로 죽었다. 익명으로 낼 수 있는 500이었다(실측).
    r = 클라.post("/api/auth/login",
                 content='{"email":NaN,"password":"password123"}'.encode(),
                 headers=제이슨)
    assert r.status_code == 422, r.text
    r2 = 클라.post("/api/auth/refresh",
                  content='{"refresh_token":NaN}'.encode(), headers=제이슨)
    assert r2.status_code == 422, r2.text


def test_Infinity가_든_요청은_422다(클라):
    r = 클라.post("/api/auth/login",
                 content='{"email":"a@example.com","password":Infinity}'.encode(),
                 headers=제이슨)
    assert r.status_code == 422, r.text
    r2 = 클라.post("/api/auth/signup", content=(가입 % "-Infinity").encode(),
                  headers=제이슨)
    assert r2.status_code == 422, r2.text


def test_고아_서로게이트가_든_요청은_422다(클라):
    # 왜: JSON 이스케이프 "\ud800"은 짝 없는 서로게이트라 utf-8로 인코딩할 수 없다.
    #     거부한 입력을 응답에 되싣으면 UnicodeEncodeError로 500이 났다(실측).
    r = 클라.post("/api/auth/signup", content=(가입 % r'"a\ud800b"').encode(),
                 headers=제이슨)
    assert r.status_code == 422, r.text
    나쁜메일 = r'{"email":"a\ud800@example.com","password":"password123"}'
    r2 = 클라.post("/api/auth/login", content=나쁜메일.encode(), headers=제이슨)
    assert r2.status_code == 422, r2.text
    # 왜 본문까지 읽는가: 500이 아니라도 서로게이트가 msg에 남으면 렌더링에서 죽는다.
    assert "\ud800" not in r.text and "\ud800" not in r2.text


def test_검증_오류는_입력값을_되돌려주지_않는다(클라):
    # 왜: F4의 표시명 검증이 422를 내면 기본 처리기는 거부한 이름과 ctx를 그대로
    #     싣는다. 본문은 type·loc·msg만 담아 입력을 되비추지 않아야 한다.
    r = 클라.post("/api/auth/signup", content=(가입 % '"   "').encode(),
                 headers=제이슨)
    assert r.status_code == 422, r.text
    항목들 = r.json()["detail"]
    assert 항목들 and all(set(항목) == {"type", "loc", "msg"} for 항목 in 항목들)
    assert "input" not in r.text and "ctx" not in r.text
    긴것 = 클라.post("/api/auth/signup",
                  content=(가입 % ('"' + "\ufb2c" * 22 + '"')).encode(),
                  headers=제이슨)
    assert 긴것.status_code == 422, 긴것.text
    assert all(set(항목) == {"type", "loc", "msg"} for 항목 in 긴것.json()["detail"])


def test_없는_경로_변수도_422로만_끝난다(클라):
    # 왜: 처리기가 loc의 정수 인덱스나 문자열 아닌 값을 만나도 죽지 않아야 한다.
    r = 클라.post("/api/auth/login", content=b'{"email":[1,2],"password":1}',
                 headers=제이슨)
    assert r.status_code == 422, r.text
    assert all(set(항목) == {"type", "loc", "msg"} for 항목 in r.json()["detail"])


# ---------- 기동 시 모델 적재와 헬스체크 ----------

def test_헬스체크가_올라온_모델_수를_준다(모델클라):
    from game.serving import store
    클라, 등록수 = 모델클라
    r = 클라.get("/api/health")
    assert r.status_code == 200, r.text
    몸체 = r.json()
    assert 몸체["status"] == "ok" and 몸체["rules_fp"] == "47c403568aa3"
    # 왜: 레지스트리에 행이 있어도 파일이 깨지면 조용히 빠진다(F6). 배포 뒤 이
    #     숫자를 보면 몇 개가 실제로 서빙 중인지 요청 한 번으로 안다.
    assert 몸체["models_loaded"] == len(store.serving()) == 등록수 > 0


def test_모델이_없어도_경고만_남기고_뜬다(tmp_path, monkeypatch, caplog):
    _환경(tmp_path, monkeypatch)
    from game.main import create_app
    with caplog.at_level("WARNING"):
        with TestClient(create_app()) as c:
            assert c.get("/api/health").json()["models_loaded"] == 0
    # 왜 경고만인가: 레지스트리가 비어 있는 것은 아직 등록 전이라는 뜻이다.
    #     개발 중 첫 기동을 막을 이유가 없다.
    assert any("모델" in 기록.message for 기록 in caplog.records)
    _전역비우기()
    reset_engine()
    get_settings.cache_clear()


def test_모델이_전부_깨지면_기동에_실패한다(tmp_path, monkeypatch):
    _환경(tmp_path, monkeypatch)
    from game.serving import store
    임시모델 = tmp_path / "models"
    임시모델.mkdir()
    for 이름 in ["rl_mc.npz", "rf_random_20.npz"]:
        shutil.copy(ROOT / "models" / 이름, 임시모델 / 이름)
    monkeypatch.setattr(store, "models_dir", 임시모델)
    assert _모델등록(임시모델) == 2
    for npz in 임시모델.glob("*.npz"):
        npz.write_bytes(b"not an npz")
    from game.main import create_app
    # 왜 기동 실패인가: 행이 있는데 하나도 못 올리면 서버가 "상대 없음"으로 조용히
    #     뜬다. RuntimeError를 삼키면 첫 손님이 500을 맞을 때까지 아무도 모른다.
    with pytest.raises(RuntimeError):
        with TestClient(create_app()):
            pass
    _전역비우기()
    reset_engine()
    get_settings.cache_clear()
