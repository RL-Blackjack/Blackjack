"""이 파일은 게임 서버 설정이 환경변수에서 올바르게 읽히는지 확인한다.
입력: 환경변수.
출력: 기본값·재정의·비밀키 검증에 대한 pytest 결과.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.config import DEFAULT_SQLITE_URL, Settings, get_settings  # noqa: E402


def test_기본값이_설계서와_같다(monkeypatch):
    monkeypatch.delenv("BJ_DATABASE_URL", raising=False)
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    s = get_settings()
    assert s.database_url == DEFAULT_SQLITE_URL
    assert s.access_token_minutes == 15
    assert s.refresh_token_days == 14
    assert s.login_rate_per_minute == 10


def test_환경변수로_DB를_바꿀_수_있다(monkeypatch):
    monkeypatch.setenv("BJ_DATABASE_URL", "postgresql+psycopg://u:p@localhost/bj")
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    assert get_settings().database_url.startswith("postgresql")


def test_비밀키가_짧으면_예외(monkeypatch):
    # 왜: 짧은 비밀키는 JWT 위조를 쉽게 만든다. 집 PC로 서비스하는 이상
    #     실수로 "secret" 같은 값을 넣고 배포하는 일을 막아야 한다.
    monkeypatch.setenv("BJ_JWT_SECRET", "short")
    get_settings.cache_clear()
    with pytest.raises(ValueError):
        get_settings()


def test_비밀키가_없으면_예외(monkeypatch):
    monkeypatch.delenv("BJ_JWT_SECRET", raising=False)
    get_settings.cache_clear()
    with pytest.raises(ValueError):
        get_settings()


def test_설정은_한_번만_읽는다(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_모델_폴더가_저장소_안을_가리킨다(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    s = get_settings()
    assert s.models_dir.name == "models"
    assert s.artifacts_dir.name == "artifacts"
