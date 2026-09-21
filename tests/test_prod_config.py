"""이 파일은 BJ_ENV=prod 가 SQLite·/docs 를 막고 PG 풀 크기가 정해진 값인지 확인한다.
입력: 환경변수 조합.
출력: ValueError·404·엔진 풀 설정에 대한 pytest 결과.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.config import get_settings  # noqa: E402
from game.db import make_engine, reset_engine  # noqa: E402

비밀 = "test-secret-at-least-32-characters-long"


@pytest.fixture(autouse=True)
def 설정초기화(monkeypatch):
    monkeypatch.delenv("BJ_ENV", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_prod는_SQLite를_거부한다(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", 비밀)
    monkeypatch.setenv("BJ_ENV", "prod")
    monkeypatch.delenv("BJ_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="postgresql"):
        get_settings()


def test_prod는_psycopg_URL만_받는다(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", 비밀)
    monkeypatch.setenv("BJ_ENV", "prod")
    monkeypatch.setenv("BJ_DATABASE_URL", "postgresql://bj:pw@db/bj")
    with pytest.raises(ValueError, match="psycopg"):
        get_settings()
    get_settings.cache_clear()
    monkeypatch.setenv("BJ_DATABASE_URL", "postgresql+psycopg://bj:pw@db/bj")
    assert get_settings().is_prod


def test_모르는_ENV_값은_거부한다(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", 비밀)
    monkeypatch.setenv("BJ_ENV", "production")
    with pytest.raises(ValueError, match="BJ_ENV"):
        get_settings()


def test_dev가_기본이고_SQLite를_받는다(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", 비밀)
    monkeypatch.delenv("BJ_DATABASE_URL", raising=False)
    설정 = get_settings()
    assert setting_ok(설정)


def setting_ok(설정) -> bool:
    return (설정.env == "dev" and not 설정.is_prod
            and 설정.database_url.startswith("sqlite:///"))


def test_PG_엔진은_풀_크기가_정해져_있다():
    엔진 = make_engine("postgresql+psycopg://bj:pw@localhost:1/bj")   # 접속하지 않는다
    try:
        assert (엔진.pool.size(), 엔진.pool._max_overflow, 엔진.pool._timeout) == (5, 5, 10)
    finally:
        엔진.dispose()


def test_prod에서는_docs가_없다(monkeypatch, tmp_path):
    monkeypatch.setenv("BJ_JWT_SECRET", 비밀)
    monkeypatch.setenv("BJ_ENV", "dev")
    monkeypatch.setenv("BJ_DATABASE_URL", f"sqlite:///{tmp_path / 'a.db'}")
    reset_engine()
    from game.main import create_app
    with TestClient(create_app()) as c:
        assert c.get("/docs").status_code == 200
    get_settings.cache_clear()
    monkeypatch.setenv("BJ_ENV", "prod")
    monkeypatch.setenv("BJ_DATABASE_URL", "postgresql+psycopg://bj:pw@localhost:1/bj")
    # 왜 lifespan 을 안 타는가: 접속할 PG 가 없다. 앱 객체의 문서 경로만 본다.
    app = create_app()
    assert app.docs_url is None and app.openapi_url is None and app.redoc_url is None
    reset_engine()


def test_비밀키_없이도_game_main을_import할_수_있다():
    # 왜: game/main.py 는 모듈 끝에서 app = create_app() 을 실행한다. 여기서 설정을 읽으면
    #   테스트 수집 단계에서 통째로 죽는다(검토에서 잡힌 함정).
    환경 = {k: v for k, v in os.environ.items() if k not in ("BJ_JWT_SECRET", "BJ_ENV")}
    환경["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT)])
    환경["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, "-c", "import game.main; print('ok')"],
                       capture_output=True, text=True, env=환경, cwd=ROOT)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr
