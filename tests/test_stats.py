"""이 파일은 전적 집계가 손계산과 일치하는지 확인한다.
입력: 손으로 심은 게임·라운드·결정 행들.
출력: 승률·일치율·EV 손실 집계에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.db import make_engine  # noqa: E402
from game.models import Base, Decision, Game, Round, User  # noqa: E402
from game.stats import (  # noqa: E402
    MIN_DECISIONS_FOR_RANK,
    recent_games,
    user_stats,
)

FP = "47c403568aa3"
지금 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def 세션():
    e = make_engine("sqlite://")      # 외래키가 켜진 서버와 같은 엔진
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s


def 사용자(s, 이름="지우"):
    u = User(email=f"{이름}@b.com", display_name=이름, auth_provider="local",
             password_hash="x", created_at=지금)
    s.add(u)
    s.commit()
    return u


def 게임심기(s, u, 라운드들, *, fp=FP, 순서=0, 열린=None):
    """라운드들 = [(net, [(고른것, 정답, 손실), ...]), ...]
    열린 = 끝나지 않은 마지막 라운드에서 이미 내린 결정들(None이면 열린 라운드 없음).
    """
    g = Game(user_id=u.id, rules_fp=fp, started_at=지금 + timedelta(minutes=순서),
             net_result=sum(n for n, _ in 라운드들))
    s.add(g)
    s.flush()
    전부 = [(net, 결정들, False) for net, 결정들 in 라운드들]
    if 열린 is not None:
        전부.append((0.0, 열린, True))  # 진행 중인 라운드는 net이 아직 0.0이다
    for i, (net, 결정들, 열림) in enumerate(전부, start=1):
        r = Round(game_id=g.id, user_id=u.id, round_no=i, seed=1000 + i, actions="0",
                  is_open=열림, dealer_up=7, dealer_cards="" if 열림 else "7,10",
                  net=net)
        s.add(r)
        s.flush()
        for j, (고른것, 정답, 손실) in enumerate(결정들):
            s.add(Decision(round_id=r.id, seq=j, total=16, is_soft=0, dealer_up=7,
                           can_double=1, can_split=0, split_depth=0,
                           action_taken=고른것, dp_optimal_action=정답,
                           dp_ev_loss=손실, ai_action=정답, created_at=지금))
    s.commit()
    return g


def test_아무것도_안_뒀으면_전부_0이다(세션):
    u = 사용자(세션)
    통계 = user_stats(세션, u.id, rules_fp=FP)
    assert (통계.games, 통계.rounds, 통계.decisions) == (0, 0, 0)
    assert 통계.win_rate == pytest.approx(0.0)
    assert 통계.agreement == pytest.approx(0.0)
    assert 통계.rank_eligible is False


def test_승패무를_손계산과_맞춘다(세션):
    u = 사용자(세션)
    게임심기(세션, u, [(1.0, [(0, 0, 0.0)]), (-1.0, [(0, 0, 0.0)]),
                   (0.0, [(0, 0, 0.0)]), (1.5, [(0, 0, 0.0)])])
    통계 = user_stats(세션, u.id, rules_fp=FP)
    assert (통계.wins, 통계.losses, 통계.pushes) == (2, 1, 1)
    assert 통계.rounds == 4
    assert 통계.win_rate == pytest.approx(2 / 4)
    assert 통계.net_result == pytest.approx(1.5)


def test_일치율을_손계산과_맞춘다(세션):
    u = 사용자(세션)
    # 결정 5개 중 3개가 정답 -> 0.6
    게임심기(세션, u, [(1.0, [(0, 0, 0.0), (1, 1, 0.0), (1, 0, 0.032),
                         (2, 0, 0.041), (0, 0, 0.0)])])
    통계 = user_stats(세션, u.id, rules_fp=FP)
    assert 통계.decisions == 5
    assert 통계.agreement == pytest.approx(3 / 5)
    assert 통계.ev_loss_total == pytest.approx(0.073)
    assert 통계.ev_loss_per_decision == pytest.approx(0.073 / 5)


def test_규칙이_다른_게임은_섞이지_않는다(세션):
    u = 사용자(세션)
    게임심기(세션, u, [(1.0, [(0, 0, 0.0)])], fp=FP)
    게임심기(세션, u, [(1.0, [(1, 0, 0.9)])], fp="ffffffffffff", 순서=1)
    # 왜: 규칙이 바뀌면 DP 정답 자체가 달라진다. 섞어 세면 일치율이 거짓말을 한다.
    통계 = user_stats(세션, u.id, rules_fp=FP)
    assert 통계.games == 1
    assert 통계.decisions == 1
    assert 통계.agreement == pytest.approx(1.0)


def test_남의_기록은_안_섞인다(세션):
    a, b = 사용자(세션, "에이"), 사용자(세션, "비")
    게임심기(세션, a, [(1.0, [(0, 0, 0.0)])])
    게임심기(세션, b, [(-1.0, [(1, 0, 0.5)] * 3)])
    통계 = user_stats(세션, a.id, rules_fp=FP)
    assert (통계.games, 통계.decisions, 통계.wins) == (1, 1, 1)


def test_결정이_200개_미만이면_랭킹_자격이_없다(세션):
    u = 사용자(세션)
    게임심기(세션, u, [(1.0, [(0, 0, 0.0)] * (MIN_DECISIONS_FOR_RANK - 1))])
    # 왜 200개인가: 설계서 §5.3이 정했다. 표본이 적으면 일치율도 흔들린다.
    assert user_stats(세션, u.id, rules_fp=FP).rank_eligible is False


def test_결정이_200개_이상이면_랭킹_자격이_생긴다(세션):
    u = 사용자(세션)
    게임심기(세션, u, [(1.0, [(0, 0, 0.0)] * MIN_DECISIONS_FOR_RANK)])
    assert user_stats(세션, u.id, rules_fp=FP).rank_eligible is True


def test_최근_게임을_새것부터_준다(세션):
    u = 사용자(세션)
    for i in range(5):
        게임심기(세션, u, [(1.0, [(0, 0, 0.0)])], 순서=i)
    목록 = recent_games(세션, u.id, rules_fp=FP, limit=3)
    assert len(목록) == 3
    assert [g.started_at for g in 목록] == sorted(
        [g.started_at for g in 목록], reverse=True)
    assert 목록[0].rounds == 1
    assert 목록[0].agreement == pytest.approx(1.0)


def test_라운드가_없는_게임도_목록에_나온다(세션):
    u = 사용자(세션)
    세션.add(Game(user_id=u.id, rules_fp=FP, started_at=지금, net_result=0.0))
    세션.commit()
    목록 = recent_games(세션, u.id, rules_fp=FP)
    assert len(목록) == 1
    assert 목록[0].rounds == 0
    # 왜: 결정이 0개일 때 0으로 나누면 터진다. 방금 만든 게임이 흔한 경우다.
    assert 목록[0].agreement == pytest.approx(0.0)


def test_열린_라운드는_승패무와_라운드_수에서_빠진다(세션):
    u = 사용자(세션)
    게임심기(세션, u, [(1.0, [(0, 0, 0.0)]), (0.0, [(0, 0, 0.0)]),
                   (-1.0, [(0, 0, 0.0)])], 열린=[(1, 0, 0.2)])
    통계 = user_stats(세션, u.id, rules_fp=FP)
    # 왜: 진행 중인 라운드는 net이 0.0이라 무승부로 세어졌다. 버린 라운드가
    #   쌓이면 승률이 끝없이 내려간다.
    assert (통계.rounds, 통계.wins, 통계.losses, 통계.pushes) == (3, 1, 1, 1)
    assert 통계.win_rate == pytest.approx(1 / 3)
    # 왜 결정은 전부 세는가: 정답과 손실은 결정 시점에 확정된다. 빼면 오답을 낸 뒤
    #   라운드를 버려 일치율에서 지울 수 있다.
    assert 통계.decisions == 4
    assert 통계.agreement == pytest.approx(3 / 4)
    assert 통계.ev_loss_total == pytest.approx(0.2)
    [줄] = recent_games(세션, u.id, rules_fp=FP)
    assert 줄.rounds == 3
    assert 줄.agreement == pytest.approx(3 / 4)


def test_열린_라운드만_있는_게임도_게임_수에는_든다(세션):
    u = 사용자(세션)
    게임심기(세션, u, [(1.0, [(0, 0, 0.0)])])
    방금것 = 게임심기(세션, u, [], 순서=1, 열린=[])
    # 왜: 열린 라운드 조건을 WHERE에 두면 이 게임의 조인 행이 통째로 빠져
    #   게임 수가 2에서 1이 된다. 조건은 조인의 ON 절에 있어야 한다.
    통계 = user_stats(세션, u.id, rules_fp=FP)
    assert (통계.games, 통계.rounds, 통계.pushes) == (2, 1, 0)
    목록 = {줄.game_id: 줄.rounds for 줄 in recent_games(세션, u.id, rules_fp=FP)}
    assert len(목록) == 2
    assert 목록[방금것.id] == 0


def test_최근_게임은_현재_규칙의_게임만_준다(세션):
    u = 사용자(세션)
    지금것 = 게임심기(세션, u, [(1.0, [(0, 0, 0.0)])])
    게임심기(세션, u, [(1.0, [(1, 0, 0.9)])], fp="ffffffffffff", 순서=1)
    # 왜: 규칙이 바뀐 옛 게임은 열면 409(rules_changed)라 이어 둘 수 없고, 일치율도
    #   다른 DP 정답 기준이다. 더 최근 게임이라도 목록에 나오면 안 된다.
    목록 = recent_games(세션, u.id, rules_fp=FP)
    assert [줄.game_id for 줄 in 목록] == [지금것.id]


def _VM단계수(세션, 조회):
    """조회를 도는 동안 SQLite 가상 머신이 실행한 명령 수와 조회 결과."""
    원시 = 세션.connection().connection.driver_connection
    수 = [0]

    def 세기():
        수[0] += 1
        return 0  # 0이 아니면 SQLite가 조회를 중단한다

    원시.set_progress_handler(세기, 1)
    try:
        결과 = 조회()
    finally:
        원시.set_progress_handler(None, 1)
    return 수[0], 결과


def test_최근_게임_집계가_남의_게임을_읽지_않는다(세션):
    나, 남 = 사용자(세션, "나"), 사용자(세션, "남")
    게임심기(세션, 나, [(1.0, [(0, 0, 0.0)])])
    전, 전결과 = _VM단계수(세션, lambda: recent_games(세션, 나.id, rules_fp=FP))
    for i in range(10):
        게임심기(세션, 남, [(-1.0, [(1, 0, 0.5)] * 5)] * 10, 순서=i)
    후, 후결과 = _VM단계수(세션, lambda: recent_games(세션, 나.id, rules_fp=FP))
    # 왜 결과가 아니라 일의 양을 보는가: 서브쿼리가 모든 사람의 라운드를 묶어도
    #   바깥 조인이 버리므로 결과는 같다. 그 대신 남의 결정 500개를 훑는 만큼
    #   명령 수가 수십 배로 늘고, 사용자가 늘수록 내 목록이 느려진다(M3).
    assert 후결과 == 전결과
    assert 후 < 전 * 2, (전, 후)


def test_전적_API가_숫자를_그대로_준다(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from game.config import get_settings
    from game.db import create_schema, make_engine, make_session_factory, reset_engine
    from game.deps import login_limiter, signup_limiter
    from game.serving import store

    monkeypatch.setenv("BJ_DATABASE_URL", f"sqlite:///{tmp_path / 's.db'}")
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    store.reset()     # 왜: 전역 모델 저장소가 앞 테스트의 DB 내용을 들고 있을 수 있다
    엔진 = make_engine(f"sqlite:///{tmp_path / 's.db'}")
    create_schema(엔진)
    login_limiter.reset()
    signup_limiter.reset()  # 왜: 앞 파일의 가입이 1분 창에 남으면 여기 가입이 429가 된다

    from game.main import create_app
    with TestClient(create_app()) as 클라:
        r = 클라.post("/api/auth/signup", json={"email": "a@b.com",
                                              "password": "hunter2!!",
                                              "display_name": "지우"})
        머리 = {"Authorization": f"Bearer {r.json()['access_token']}"}
        사용자번호 = 클라.get("/api/auth/me", headers=머리).json()["id"]

        with make_session_factory(엔진)() as s:
            게임심기(s, s.get(User, 사용자번호),
                  [(1.0, [(0, 0, 0.0), (1, 0, 0.05)])])
            # 왜: 규칙이 다른 옛 게임은 목록에서 빠져야 한다(H2를 API 호출부까지 확인)
            게임심기(s, s.get(User, 사용자번호), [(1.0, [])], fp="ffffffffffff", 순서=1)

        통계 = 클라.get("/api/me/stats", headers=머리)
        assert 통계.status_code == 200, 통계.text
        몸체 = 통계.json()
        assert 몸체["decisions"] == 2
        assert 몸체["agreement"] == pytest.approx(0.5)
        assert 몸체["ev_loss_total"] == pytest.approx(0.05)

        목록 = 클라.get("/api/me/games", headers=머리)
        assert 목록.status_code == 200
        assert len(목록.json()) == 1

    엔진.dispose()
    reset_engine()
    get_settings.cache_clear()


def test_로그인하지_않으면_전적을_못_본다(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from game.config import get_settings
    from game.db import reset_engine
    from game.serving import store

    monkeypatch.setenv("BJ_DATABASE_URL", f"sqlite:///{tmp_path / 'n.db'}")
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    store.reset()     # 왜: 전역 모델 저장소가 앞 테스트의 DB 내용을 들고 있을 수 있다
    from game.main import create_app
    with TestClient(create_app()) as 클라:
        assert 클라.get("/api/me/stats").status_code == 401
    reset_engine()
    get_settings.cache_clear()
