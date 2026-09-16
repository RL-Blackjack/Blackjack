"""이 파일은 DB 스키마가 설계서대로 만들어지고 제약이 걸리는지 확인한다.
입력: 메모리 SQLite.
출력: 테이블·컬럼·유니크·외래키에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.db import make_engine  # noqa: E402
from game.models import (  # noqa: E402
    Base,
    Decision,
    Game,
    ModelRegistry,
    RefreshToken,
    Round,
    User,
)

지금 = datetime.now(timezone.utc)


@pytest.fixture
def 세션():
    # 왜 create_engine이 아니라 make_engine인가: SQLite는 외래키를 기본으로 무시한다.
    #     서버가 쓰는 유일한 엔진 공장을 거쳐야 PRAGMA foreign_keys=ON 이 켜지고,
    #     그래야 이 파일이 배포(PostgreSQL)와 같은 제약 아래에서 돈다.
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def 사용자(세션, email="a@b.com"):
    u = User(email=email, display_name="지우", auth_provider="local",
             password_hash="$argon2id$dummy")
    세션.add(u)
    세션.commit()
    return u


def test_테이블_여섯_개가_만들어진다(세션):
    이름들 = set(inspect(세션.get_bind()).get_table_names())
    assert {"users", "refresh_tokens", "model_registry",
            "games", "rounds", "decisions"} <= 이름들


def test_이메일은_유일하다(세션):
    사용자(세션, "dup@b.com")
    세션.add(User(email="dup@b.com", display_name="다른사람",
                 auth_provider="local", password_hash="x"))
    with pytest.raises(IntegrityError):
        세션.commit()


def test_구글_계정은_비밀번호가_없어도_된다(세션):
    u = User(email="g@b.com", display_name="구글유저",
             auth_provider="google", google_sub="sub-123")
    세션.add(u)
    세션.commit()
    assert u.password_hash is None
    assert u.id is not None


def test_결정은_상태_여섯_필드를_그대로_담는다(세션):
    u = 사용자(세션)
    g = Game(user_id=u.id, rules_fp="47c403568aa3", started_at=지금)
    세션.add(g)
    세션.commit()
    r = Round(game_id=g.id, round_no=1, seed=12345, actions="1,0",
              is_open=False, dealer_up=10, dealer_cards="10,7", net=-1.0)
    세션.add(r)
    세션.commit()
    d = Decision(round_id=r.id, seq=0, total=16, is_soft=0, dealer_up=10,
                 can_double=1, can_split=0, split_depth=0,
                 action_taken=1, dp_optimal_action=1, dp_ev_loss=0.0,
                 ai_action=1, created_at=지금)
    세션.add(d)
    세션.commit()

    실린값 = 세션.scalar(select(Decision))
    assert (실린값.total, 실린값.is_soft, 실린값.dealer_up,
            실린값.can_double, 실린값.can_split, 실린값.split_depth) == (16, 0, 10, 1, 0, 0)
    assert 실린값.dp_ev_loss == pytest.approx(0.0)


def test_라운드는_시드와_행동목록을_담는다(세션):
    # 왜: 서버는 카드를 저장하지 않고 시드로 재현한다. 이 두 칸이 라운드의 전부다.
    u = 사용자(세션)
    g = Game(user_id=u.id, rules_fp="47c403568aa3", started_at=지금)
    세션.add(g)
    세션.commit()
    세션.add(Round(game_id=g.id, round_no=1, seed=2**40 + 7, dealer_up=9))
    세션.commit()

    r = 세션.scalar(select(Round))
    assert r.seed == 2**40 + 7          # 큰 시드가 잘려서는 안 된다
    assert r.actions == ""              # 아직 아무 행동도 없다
    assert r.is_open is True
    assert r.dealer_cards == ""         # 라운드가 끝나기 전이다
    assert r.net == pytest.approx(0.0)


def test_게임에_규칙_지문이_박힌다(세션):
    # 왜: 규칙이 바뀌면 옛 기록과 새 기록을 섞어 집계하면 안 된다.
    u = 사용자(세션)
    g = Game(user_id=u.id, rules_fp="47c403568aa3", started_at=지금)
    세션.add(g)
    세션.commit()
    assert 세션.scalar(select(Game)).rules_fp == "47c403568aa3"


def test_없는_사용자의_게임은_만들_수_없다(세션):
    세션.add(Game(user_id=99999, rules_fp="x", started_at=지금))
    with pytest.raises(IntegrityError):
        세션.commit()


def test_리프레시_토큰은_해시로만_저장된다(세션):
    u = 사용자(세션)
    t = RefreshToken(user_id=u.id, token_hash="sha256-of-token",
                     expires_at=지금 + timedelta(days=14))
    세션.add(t)
    세션.commit()
    컬럼 = {c["name"] for c in inspect(세션.get_bind()).get_columns("refresh_tokens")}
    # 왜: 토큰 원문을 저장하면 DB가 유출될 때 그대로 로그인에 쓰인다.
    assert "token" not in 컬럼
    assert "token_hash" in 컬럼


def test_모델_레지스트리가_서빙_여부를_담는다(세션):
    m = ModelRegistry(name="rl_mc_headline", family="rl", rules_fp="47c403568aa3",
                      artifact_path="artifacts/x.npz", artifact_sha256="a" * 64,
                      exact_ev=-0.008822, is_serving=True)
    세션.add(m)
    세션.commit()
    assert 세션.scalar(select(ModelRegistry)).is_serving is True


def test_모델_이름은_유일하다(세션):
    for _ in range(2):
        세션.add(ModelRegistry(name="dup", family="rl", rules_fp="x",
                              artifact_path="p", artifact_sha256="b" * 64,
                              exact_ev=-0.1, is_serving=True))
    with pytest.raises(IntegrityError):
        세션.commit()
