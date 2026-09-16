"""이 파일은 DB 스키마가 설계서대로 만들어지고 칸이 값을 제대로 담는지 확인한다.
입력: 메모리 SQLite(PG 쪽 시각 정규화는 타입 메서드를 직접 부른다).
출력: 테이블·컬럼·상대 모델 해시·UTC 시각에 대한 pytest 결과(제약은 test_db_constraints.py).
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import DateTime, func, inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, StatementError
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
    UTCDateTime,
)

지금 = datetime.now(timezone.utc)
# 왜 ZoneInfo가 아닌 고정 오프셋인가: 윈도우는 tzdata 패키지가 없으면 ZoneInfo를
#   못 만든다. 정규화는 astimezone만 쓰므로 고정 오프셋으로도 같은 경로를 탄다.
KST = timezone(timedelta(hours=9), "KST")


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
    r = Round(game_id=g.id, user_id=u.id, round_no=1, seed=12345, actions="1,0",
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
    세션.add(Round(game_id=g.id, user_id=u.id, round_no=1, seed=2**40 + 7, dealer_up=9))
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


def test_게임은_상대_모델의_파일_해시를_담는다(세션):
    # 왜: 레지스트리 행은 같은 이름으로 다시 등록하면 해시가 덮어쓰인다. 게임에 박힌
    #   해시로만 decisions.ai_action 이 어느 정책 파일에서 나왔는지 가려낸다.
    u = 사용자(세션)
    해시 = "0123456789abcdef" * 4
    세션.add_all([
        Game(user_id=u.id, rules_fp="x", started_at=지금, opponent_artifact_sha256=해시),
        Game(user_id=u.id, rules_fp="x", started_at=지금),  # 상대 모델이 없는 게임
    ])
    세션.commit()
    세션.expire_all()
    assert 세션.scalars(select(Game.opponent_artifact_sha256).order_by(Game.id)).all() == [해시, None]
    # 왜 반영된 스키마도 보는가: SQLite는 길이를 강제하지 않는다. PG의 VARCHAR(64)는
    #   이 선언에서 나오므로 선언이 64자·NULL 허용인지 직접 본다.
    칸 = {c["name"]: c for c in inspect(세션.get_bind()).get_columns("games")}
    assert 칸["opponent_artifact_sha256"]["nullable"] is True
    assert 칸["opponent_artifact_sha256"]["type"].length == 64


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


def test_시각_칸_여섯_개가_모두_UTC_타입이다():
    # 왜 python_type이 아닌가: TypeDecorator는 python_type을 위임하지 않아 예외를 낸다.
    칸들 = [c for t in Base.metadata.sorted_tables for c in t.columns
            if isinstance(c.type, (DateTime, UTCDateTime))]
    assert len(칸들) == 6
    assert all(isinstance(c.type, UTCDateTime) for c in 칸들)


def test_시각은_UTC_aware로_되읽힌다(세션):
    u = 사용자(세션)
    세션.add(Game(user_id=u.id, rules_fp="x",
                 started_at=datetime(2026, 9, 16, 21, 0, tzinfo=KST)))
    세션.commit()
    # 왜 원문을 보는가: SQLite는 오프셋을 버리고 벽시계만 쓴다. UTC로 바꿔 넣지
    #   않으면 문자열 정렬·비교가 9시간 어긋난다.
    원문 = 세션.scalar(text("SELECT started_at FROM games"))
    assert str(원문).startswith("2026-09-16 12:00:00")
    세션.expire_all()
    읽은값 = 세션.scalar(select(Game.started_at))
    assert 읽은값 == datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    assert 읽은값.tzinfo is timezone.utc


def test_naive_시각은_저장을_거부한다(세션):
    # 왜 조용히 UTC로 보지 않는가: naive 값은 호출부 버그다. 드러나야 고친다.
    u = 사용자(세션)
    세션.add(Game(user_id=u.id, rules_fp="x", started_at=datetime(2026, 9, 16, 12, 0)))
    with pytest.raises(StatementError) as 오류:
        세션.commit()
    assert isinstance(오류.value.orig, ValueError)
    세션.rollback()
    assert 세션.scalar(select(func.count()).select_from(Game)) == 0


def test_PG가_KST로_돌려줘도_UTC로_정규화된다():
    # 왜: psycopg3는 연결의 TimeZone(예: Asia/Seoul) 기준 aware 값을 준다.
    타입, 방언 = UTCDateTime(), postgresql.dialect()
    읽은값 = 타입.process_result_value(datetime(2026, 9, 16, 21, 0, tzinfo=KST), 방언)
    assert 읽은값 == datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
    assert 읽은값.tzinfo is timezone.utc
    넣을값 = 타입.process_bind_param(datetime(2026, 9, 16, 21, 0, tzinfo=KST), 방언)
    assert 넣을값.tzinfo is timezone.utc and 넣을값.hour == 12
    assert 타입.process_result_value(None, 방언) is None
