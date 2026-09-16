"""이 파일은 엔진·세션·스키마 생성이 SQLite에서 제대로 도는지 확인한다.
입력: 임시 SQLite 파일.
출력: 스키마 생성·삭제, 외래키 강제, 세션 의존성에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.config import get_settings  # noqa: E402
from game.db import (  # noqa: E402
    create_schema,
    drop_schema,
    get_session,
    make_engine,
    make_session_factory,
    reset_engine,
)
from game.models import Game, User  # noqa: E402

지금 = datetime.now(timezone.utc)


@pytest.fixture
def 엔진(tmp_path):
    e = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    create_schema(e)
    yield e
    e.dispose()


def test_스키마를_만들면_표가_생긴다(엔진):
    assert "users" in inspect(엔진).get_table_names()


def test_적용과_롤백이_깨끗하게_반복된다(엔진):
    # 왜: 설계서 §10의 "마이그레이션이 깨끗하게 적용·롤백된다"를 이렇게 확인한다.
    #     Alembic 대신 create_all/drop_all을 쓰기로 했기 때문이다.
    for _ in range(3):
        drop_schema(엔진)
        assert inspect(엔진).get_table_names() == []
        create_schema(엔진)
        assert len(inspect(엔진).get_table_names()) == 6


def test_두_번_만들어도_터지지_않는다(엔진):
    create_schema(엔진)
    assert len(inspect(엔진).get_table_names()) == 6


def test_SQLite에서_외래키가_강제된다(엔진):
    # 왜: SQLite는 기본적으로 외래키를 무시한다. 훅으로 켜지 않으면 테스트는
    #     통과하는데 배포(PostgreSQL)에서만 제약 위반이 터진다.
    공장 = make_session_factory(엔진)
    with 공장() as s:
        assert s.execute(text("PRAGMA foreign_keys")).scalar() == 1
        s.add(Game(user_id=424242, rules_fp="x", started_at=지금))
        with pytest.raises(IntegrityError):
            s.commit()


def test_세션_공장은_쓸_때마다_새_세션을_준다(엔진):
    공장 = make_session_factory(엔진)
    with 공장() as a, 공장() as b:
        assert a is not b


def test_의존성이_세션을_열고_닫는다(tmp_path, monkeypatch):
    monkeypatch.setenv("BJ_DATABASE_URL", f"sqlite:///{tmp_path / 'dep.db'}")
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    try:
        열린것 = list(get_session())
        assert len(열린것) == 1
        세션 = 열린것[0]
        세션.add(User(email="dep@b.com", display_name="지우",
                     auth_provider="local", password_hash="x", created_at=지금))
        세션.commit()
        assert 세션.scalar(select(User).where(User.email == "dep@b.com")) is not None
        세션.close()
    finally:
        reset_engine()
        get_settings.cache_clear()
