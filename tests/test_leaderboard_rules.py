"""이 파일은 리더보드 집계의 정의와 공개 API 경계를 보강해서 확인한다.
입력: 손으로 심은 게임·라운드·결정과 TestClient 요청.
출력: 집계 부풀림·규칙 지문 분리·공유 캐시·limit 경계에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.db import make_engine  # noqa: E402
from game.leaderboard import TOP_N, leaderboard  # noqa: E402
from game.models import Base, Decision, Game, Round, User  # noqa: E402
from game.stats import MIN_DECISIONS_FOR_RANK  # noqa: E402

FP = "47c403568aa3"
옛FP = "ffffffffffff"
지금 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
N = MIN_DECISIONS_FOR_RANK


@pytest.fixture
def 세션():
    e = make_engine("sqlite://")      # 외래키가 켜진 서버와 같은 엔진
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s
    e.dispose()


def 사용자(s, 이름):
    """표시명만 다른 회원 한 명을 만든다."""
    u = User(email=f"{이름}@b.com", display_name=이름, auth_provider="local",
             password_hash="x", created_at=지금)
    s.add(u)
    s.flush()
    return u


def 게임(s, u, *, 순손익=0.0, fp=FP):
    """규칙 지문과 순손익을 정해 게임 한 판을 만든다."""
    g = Game(user_id=u.id, rules_fp=fp, started_at=지금, net_result=순손익)
    s.add(g)
    s.flush()
    return g


def 라운드(s, g, 회차, *, net=0.0):
    """이미 끝난 라운드 하나를 만든다."""
    r = Round(game_id=g.id, user_id=g.user_id, round_no=회차, seed=회차,
              actions="0", is_open=False, dealer_up=7, dealer_cards="7,10",
              net=net)
    s.add(r)
    s.flush()
    return r


def 결정들(s, r, 수, 맞은수, *, 손실=0.02):
    """라운드에 결정 N개를 달고 그중 앞 M개를 정답과 같게 둔다."""
    for i in range(수):
        맞음 = i < 맞은수
        s.add(Decision(round_id=r.id, seq=i, total=16, is_soft=0, dealer_up=7,
                       can_double=1, can_split=0, split_depth=0,
                       action_taken=0, dp_optimal_action=0 if 맞음 else 1,
                       dp_ev_loss=0.0 if 맞음 else 손실, ai_action=0,
                       created_at=지금))


def 쿼리_수(할일):
    """할일을 돌리는 동안 어느 엔진으로든 나간 SQL 문의 수와 결과를 준다."""
    # 왜 Engine 클래스에 거는가: 앱은 자기 엔진을 따로 만든다. 테스트가 만든
    #   엔진에만 걸면 요청이 친 쿼리를 한 건도 못 센다.
    나간것 = []

    def 세기(*_인자):
        나간것.append(1)

    event.listen(Engine, "before_cursor_execute", 세기)
    try:
        결과 = 할일()
    finally:
        event.remove(Engine, "before_cursor_execute", 세기)
    return len(나간것), 결과


@pytest.fixture
def 클라(tmp_path, monkeypatch):
    """세 사람이 심긴 파일 DB 위에 공개 API 클라이언트를 띄운다."""
    from fastapi.testclient import TestClient

    from game.config import get_settings
    from game.db import create_schema, make_session_factory, reset_engine
    from game.leaderboard import cache
    from game.main import create_app
    from game.serving import store
    from game.deps import login_limiter, signup_limiter

    주소 = f"sqlite:///{tmp_path / 'lb_rules.db'}"
    monkeypatch.setenv("BJ_DATABASE_URL", 주소)
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    # 왜 둘 다 비우는가: 캐시와 모델 저장소는 모듈 전역이라 앞 테스트 파일이
    #   남긴 값이 그대로 보인다(실행 순서에 따라 결과가 달라진다).
    cache.clear()
    store.reset()
    # 왜: 제한기는 프로세스 전역이라 앞 테스트 파일의 가입·로그인 횟수가 남는다.
    login_limiter.reset()
    signup_limiter.reset()
    엔진 = make_engine(주소)
    create_schema(엔진)
    with make_session_factory(엔진)() as s:
        for i in range(3):
            u = 사용자(s, f"사람{i}")
            g = 게임(s, u, 순손익=1.0)
            결정들(s, 라운드(s, g, 1, net=1.0), N, N - i)
        s.commit()

    with TestClient(create_app()) as 클라이언트:
        yield 클라이언트

    엔진.dispose()
    reset_engine()
    cache.clear()
    get_settings.cache_clear()


def test_한_게임에_닫힌_라운드가_둘이어도_집계가_부풀지_않는다(세션):
    u = 사용자(세션, "지우")
    g = 게임(세션, u, 순손익=2.0)
    결정들(세션, 라운드(세션, g, 1, net=1.0), N // 2, N // 2)
    결정들(세션, 라운드(세션, g, 2, net=1.0), N - N // 2, N - N // 2)
    세션.commit()
    줄 = leaderboard(세션, rules_fp=FP)
    # 왜: 게임 집계를 결정·라운드와 한 조인에 넣으면 라운드 수만큼 곱해져
    #   게임 2판, 순손익 4.0으로 보인다. 두 칸은 게임 행만 세야 한다.
    assert len(줄) == 1
    assert 줄[0].games == 1
    assert 줄[0].net_result == pytest.approx(2.0)
    assert 줄[0].decisions == N


def test_같은_사람의_옛_규칙_기록은_한_줄에_섞이지_않는다(세션):
    u = 사용자(세션, "지우")
    이번 = 게임(세션, u, 순손익=3.0)
    결정들(세션, 라운드(세션, 이번, 1, net=3.0), N, N)
    옛것 = 게임(세션, u, 순손익=100.0, fp=옛FP)
    결정들(세션, 라운드(세션, 옛것, 1, net=100.0), N, 0, 손실=0.5)
    세션.commit()
    줄 = leaderboard(세션, rules_fp=FP)
    # 왜 같은 사람으로 보는가: 사람을 나눠 심으면 옛 규칙 줄이 통째로 빠지는지만
    #   보게 된다. 지문 필터가 한쪽 집계에서 빠지면 같은 사람 줄에 섞여 들어온다.
    assert len(줄) == 1
    assert 줄[0].decisions == N
    assert 줄[0].agreement == pytest.approx(1.0)
    assert 줄[0].ev_loss_per_decision == pytest.approx(0.0)
    assert 줄[0].games == 1
    assert 줄[0].net_result == pytest.approx(3.0)


def test_공개_API는_limit이_달라도_같은_캐시를_쓴다(클라):
    첫번째 = 클라.get("/api/leaderboard?limit=1")
    assert 첫번째.status_code == 200, 첫번째.text
    횟수, 두번째 = 쿼리_수(lambda: 클라.get(f"/api/leaderboard?limit={TOP_N}"))
    assert 두번째.status_code == 200, 두번째.text
    # 왜: 캐시 키에 limit이 들어가면 인증 없는 공개 API에서 limit만 1~100으로
    #   바꿔 가며 불러 5분마다 전체 집계를 100번 일으킬 수 있다.
    assert 횟수 == 0
    assert len(첫번째.json()) == 1
    assert len(두번째.json()) == 3
    assert 두번째.json()[0] == 첫번째.json()[0]
    assert [r["rank"] for r in 두번째.json()] == [1, 2, 3]


def test_limit는_100까지_받고_101은_422다(클라):
    # 왜 100인가: 캐시는 상위 TOP_N줄만 들고 있다. 그보다 크게 받으면 잘라 줄
    #   내용이 없는데도 200을 주게 된다.
    assert TOP_N == 100
    좋음 = 클라.get("/api/leaderboard?limit=100")
    assert 좋음.status_code == 200, 좋음.text
    assert len(좋음.json()) == 3
    assert 클라.get("/api/leaderboard?limit=101").status_code == 422
