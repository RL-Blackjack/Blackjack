"""이 파일은 리더보드가 일치율로 줄 세우고 자격 미달을 빼는지 확인한다.
입력: 손으로 심은 사용자와 결정들.
출력: 순위·자격·캐시에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.leaderboard import CACHE_SECONDS, TTLCache, leaderboard  # noqa: E402
from game.db import make_engine  # noqa: E402
from game.models import Base, Decision, Game, Round, User  # noqa: E402
from game.stats import MIN_DECISIONS_FOR_RANK  # noqa: E402

FP = "47c403568aa3"
지금 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def 세션():
    e = make_engine("sqlite://")      # 외래키가 켜진 서버와 같은 엔진
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s


def 심기(s, 이름, 결정수, 맞은수, *, 손실=0.02, 순손익=0.0, fp=FP):
    """결정 N개 중 M개를 맞힌 사람을 만든다."""
    u = User(email=f"{이름}@b.com", display_name=이름, auth_provider="local",
             password_hash="x", created_at=지금)
    s.add(u)
    s.flush()
    g = Game(user_id=u.id, rules_fp=fp, started_at=지금, net_result=순손익)
    s.add(g)
    s.flush()
    r = Round(game_id=g.id, round_no=1, seed=1, actions="0", is_open=False,
              dealer_up=7, dealer_cards="7,10", net=순손익)
    s.add(r)
    s.flush()
    for i in range(결정수):
        맞음 = i < 맞은수
        s.add(Decision(round_id=r.id, seq=i, total=16, is_soft=0, dealer_up=7,
                       can_double=1, can_split=0, split_depth=0,
                       action_taken=0, dp_optimal_action=0 if 맞음 else 1,
                       dp_ev_loss=0.0 if 맞음 else 손실, ai_action=0,
                       created_at=지금))
    s.commit()
    return u


def test_아무도_없으면_빈_목록이다(세션):
    assert leaderboard(세션, rules_fp=FP) == []


def test_일치율_순으로_줄_세운다(세션):
    N = MIN_DECISIONS_FOR_RANK
    심기(세션, "중간", N, int(N * 0.7))
    심기(세션, "잘함", N, int(N * 0.9))
    심기(세션, "못함", N, int(N * 0.5))
    줄 = leaderboard(세션, rules_fp=FP)
    assert [r.display_name for r in 줄] == ["잘함", "중간", "못함"]
    assert [r.rank for r in 줄] == [1, 2, 3]
    assert 줄[0].agreement == pytest.approx(0.9)


def test_승률로_줄_세우지_않는다(세션):
    N = MIN_DECISIONS_FOR_RANK
    # 왜 이 검사인가: 설계서 §5.3의 결론이다. 운 좋은 사람이 1등을 하면 안 된다.
    심기(세션, "운좋음", N, int(N * 0.5), 순손익=50.0)
    심기(세션, "잘둠", N, int(N * 0.95), 순손익=-10.0)
    줄 = leaderboard(세션, rules_fp=FP)
    assert 줄[0].display_name == "잘둠"
    assert 줄[0].net_result < 줄[1].net_result


def test_결정_200개_미만은_빠진다(세션):
    심기(세션, "자격있음", MIN_DECISIONS_FOR_RANK, MIN_DECISIONS_FOR_RANK)
    심기(세션, "표본부족", MIN_DECISIONS_FOR_RANK - 1, MIN_DECISIONS_FOR_RANK - 1)
    줄 = leaderboard(세션, rules_fp=FP)
    assert [r.display_name for r in 줄] == ["자격있음"]


def test_한_번도_안_둔_사람은_안_나온다(세션):
    세션.add(User(email="z@b.com", display_name="구경꾼", auth_provider="local",
                 password_hash="x", created_at=지금))
    세션.commit()
    심기(세션, "선수", MIN_DECISIONS_FOR_RANK, MIN_DECISIONS_FOR_RANK)
    assert [r.display_name for r in leaderboard(세션, rules_fp=FP)] == ["선수"]


def test_규칙이_다른_기록은_안_센다(세션):
    N = MIN_DECISIONS_FOR_RANK
    심기(세션, "옛규칙", N, N, fp="ffffffffffff")
    심기(세션, "지금규칙", N, int(N * 0.6))
    assert [r.display_name for r in leaderboard(세션, rules_fp=FP)] == ["지금규칙"]


def test_개수_제한이_먹는다(세션):
    for i in range(5):
        심기(세션, f"사람{i}", MIN_DECISIONS_FOR_RANK,
            MIN_DECISIONS_FOR_RANK - i)
    assert len(leaderboard(세션, rules_fp=FP, limit=3)) == 3


def test_EV_손실도_함께_준다(세션):
    N = MIN_DECISIONS_FOR_RANK
    심기(세션, "지우", N, N - 10, 손실=0.05)
    줄 = leaderboard(세션, rules_fp=FP)[0]
    # 틀린 결정 10개 x 0.05 = 0.5, 결정당 0.5/200
    assert 줄.ev_loss_per_decision == pytest.approx(0.5 / N)


def test_캐시는_시간이_지나야_비워진다():
    캐시 = TTLCache(ttl_seconds=300.0)
    캐시.put("k", [1, 2, 3], now=1000.0)
    assert 캐시.get("k", now=1000.0) == [1, 2, 3]
    assert 캐시.get("k", now=1299.0) == [1, 2, 3]
    # 왜 5분인가: 설계서 §6.3이 정했다. 리더보드가 5분 늦는 것은 문제가 안 된다.
    assert 캐시.get("k", now=1301.0) is None
    assert CACHE_SECONDS == pytest.approx(300.0)


def test_캐시_키가_다르면_안_섞인다():
    캐시 = TTLCache(ttl_seconds=300.0)
    캐시.put("a", [1], now=0.0)
    캐시.put("b", [2], now=0.0)
    assert 캐시.get("a", now=0.0) == [1]
    assert 캐시.get("b", now=0.0) == [2]


def test_리더보드_API가_열려_있다(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from game.config import get_settings
    from game.db import create_schema, make_engine, make_session_factory, reset_engine
    from game.leaderboard import cache

    주소 = f"sqlite:///{tmp_path / 'lb.db'}"
    monkeypatch.setenv("BJ_DATABASE_URL", 주소)
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    cache.clear()
    엔진 = make_engine(주소)
    create_schema(엔진)
    with make_session_factory(엔진)() as s:
        심기(s, "지우", MIN_DECISIONS_FOR_RANK, MIN_DECISIONS_FOR_RANK - 20)

    from game.main import create_app
    with TestClient(create_app()) as 클라:
        # 왜 로그인 없이 되는가: 리더보드는 공개다. 이름과 일치율만 나가고
        #   이메일 같은 개인정보는 나가지 않는다.
        r = 클라.get("/api/leaderboard")
        assert r.status_code == 200, r.text
        줄들 = r.json()
        assert len(줄들) == 1
        assert 줄들[0]["display_name"] == "지우"
        assert "email" not in 줄들[0]
        assert 줄들[0]["rank"] == 1

    엔진.dispose()
    reset_engine()
    cache.clear()
    get_settings.cache_clear()
