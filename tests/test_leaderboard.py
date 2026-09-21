"""이 파일은 리더보드가 일치율로 줄 세우고 자격 미달을 빼는지 확인한다.
입력: 손으로 심은 사용자와 결정들.
출력: 순위·자격·캐시에 대한 pytest 결과.
"""

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from game.leaderboard import (  # noqa: E402
    CACHE_SECONDS, TTLCache, cached_leaderboard, leaderboard)
from game.db import make_engine  # noqa: E402
from game_helpers import fresh_schema, test_db_url  # noqa: E402
from game.models import Base, Decision, Game, Round, User  # noqa: E402
from game.stats import MIN_DECISIONS_FOR_RANK  # noqa: E402

FP = "47c403568aa3"
지금 = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def 세션(tmp_path):
    e = make_engine(test_db_url(tmp_path, "lb.db"))      # 외래키가 켜진 서버와 같은 엔진
    fresh_schema(e)
    with Session(e) as s:
        yield s
    e.dispose()


def 심기(s, 이름, 결정수, 맞은수, *, 손실=0.02, 순손익=0.0, fp=FP):
    """결정 N개 중 M개를 맞힌 사람을 만든다."""
    u = User(email=f"{이름}@b.com", display_name=이름, auth_provider="local",
             password_hash="x", created_at=지금)
    s.add(u)
    s.flush()
    g = Game(user_id=u.id, rules_fp=fp, started_at=지금, net_result=순손익)
    s.add(g)
    s.flush()
    r = Round(game_id=g.id, user_id=u.id, round_no=1, seed=1, actions="0",
              is_open=False, dealer_up=7, dealer_cards="7,10", net=순손익)
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


def 쿼리_수(세션, 할일):
    """할일을 돌리는 동안 DB로 나간 SQL 문의 수와 할일의 결과를 준다."""
    나간것 = []
    엔진 = 세션.get_bind()
    def 세기(*_인자):
        나간것.append(1)
    event.listen(엔진, "before_cursor_execute", 세기)
    try:
        결과 = 할일()
    finally:
        event.remove(엔진, "before_cursor_execute", 세기)
    return len(나간것), 결과


class 멈추는칸(dict):
    """스레드 A가 dict.get을 마친 직후 멈춰, 그 틈에 B가 끼어들게 하는 칸."""

    def __init__(self):
        super().__init__()
        self.a_읽음 = threading.Event()
        self.b_끝남 = threading.Event()

    def get(self, key, default=None):
        값 = dict.get(self, key, default)
        if threading.current_thread().name == "A" and not self.a_읽음.is_set():
            self.a_읽음.set()
            # 왜 기다림에 상한이 있는가: 잠금이 있으면 B는 A가 끝나야 들어온다.
            #   끝없이 기다리면 서로 막히므로 잠깐 기다린 뒤 A를 계속 보낸다.
            self.b_끝남.wait(0.3)
        return 값


def 끼어들기(b의_일):
    """A가 만료된 값을 읽고 멈춘 틈에 B가 b의_일을 한다. (스레드별 결과, 캐시)를 준다."""
    캐시 = TTLCache(ttl_seconds=300.0)
    캐시._칸 = 칸 = 멈추는칸()
    캐시.put("k", "옛값", now=0.0)
    결과 = {}

    def 돌리기(이름, 일):
        if 이름 == "B":
            칸.a_읽음.wait(5)      # A가 만료된 값을 읽고 멈출 때까지
        try:
            결과[이름] = ("ok", 일())
        except Exception as e:  # noqa: BLE001 - 어떤 예외든 결과로 남겨 비교한다
            결과[이름] = ("예외", repr(e))
        finally:
            if 이름 == "B":
                칸.b_끝남.set()

    스레드들 = [
        threading.Thread(name="A", target=돌리기,
                         args=("A", lambda: 캐시.get("k", now=1000.0))),
        threading.Thread(name="B", target=돌리기, args=("B", lambda: b의_일(캐시))),
    ]
    for t in 스레드들:
        t.start()
    for t in 스레드들:
        t.join()
    return 결과, 캐시


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


def test_캐시가_동시_만료에서_KeyError를_내지_않는다():
    # 왜: 요청은 스레드 풀에서 돈다. A가 만료 값을 읽은 뒤 B가 먼저 지우면
    #   A의 del이 KeyError로 그 요청을 500으로 만들었다(결정적 인터리빙으로 재현).
    결과, _ = 끼어들기(lambda c: c.get("k", now=1000.0))
    assert 결과 == {"A": ("ok", None), "B": ("ok", None)}


def test_만료_삭제가_끼어든_새_값을_지우지_않는다():
    # 왜: pop만으로는 A가 만료를 본 뒤 B가 넣은 새 값을 A가 지울 수 있다. 잠금으로 막는다.
    결과, 캐시 = 끼어들기(lambda c: c.put("k", "새값", now=1000.0))
    assert 결과 == {"A": ("ok", None), "B": ("ok", None)}
    assert 캐시.get("k", now=1000.0) == "새값"


def test_리더보드는_사용자_수와_무관하게_쿼리_수가_일정하다(세션):
    N = MIN_DECISIONS_FOR_RANK
    심기(세션, "사람0", N, N)
    한명, _ = 쿼리_수(세션, lambda: leaderboard(세션, rules_fp=FP))
    for i in range(1, 20):
        심기(세션, f"사람{i}", N, N - i)
    스무명, 줄 = 쿼리_수(세션, lambda: leaderboard(세션, rules_fp=FP))
    # 왜: 순손익을 사람마다 다시 세면 1+N번 친다. 인증 없는 공개 API다.
    assert len(줄) == 20
    assert 스무명 == 한명


def test_게임_수와_순손익이_같은_게임_집합을_센다(세션):
    u = 심기(세션, "지우", MIN_DECISIONS_FOR_RANK, MIN_DECISIONS_FOR_RANK, 순손익=3.0)
    결정없음 = Game(user_id=u.id, rules_fp=FP, started_at=지금, net_result=-1.0)
    옛규칙 = Game(user_id=u.id, rules_fp="ffffffffffff", started_at=지금,
               net_result=100.0)
    세션.add_all([결정없음, 옛규칙])
    세션.flush()
    # 첫 딜이 딜러 블랙잭이라 결정 없이 끝난 라운드
    세션.add(Round(game_id=결정없음.id, user_id=u.id, round_no=1, seed=2,
                  actions="", is_open=False, dealer_up=10, dealer_cards="10,1",
                  net=-1.0))
    세션.commit()
    줄 = leaderboard(세션, rules_fp=FP)[0]
    # 왜: games는 결정이 있는 게임만, net_result는 전체를 세서 두 칸이 가리키는
    #   집합이 달랐다. 둘 다 "규칙 지문이 같은 게임 전체"를 센다.
    assert 줄.games == 2
    assert 줄.net_result == pytest.approx(2.0)


def test_limit가_달라도_같은_캐시를_쓴다(세션):
    for i in range(5):
        심기(세션, f"사람{i}", MIN_DECISIONS_FOR_RANK, MIN_DECISIONS_FOR_RANK - i)
    캐시 = TTLCache()
    앞둘 = cached_leaderboard(세션, rules_fp=FP, limit=2, ttl_cache=캐시)
    횟수, 다섯 = 쿼리_수(세션, lambda: cached_leaderboard(
        세션, rules_fp=FP, limit=5, ttl_cache=캐시))
    # 왜: limit마다 키를 두면 limit=1~100을 돌며 5분마다 전체 집계를 100번
    #   일으킬 수 있다. 상위 100줄을 한 번 세어 두고 잘라 준다.
    assert 횟수 == 0
    assert [r.display_name for r in 앞둘] == ["사람0", "사람1"]
    assert [r.rank for r in 다섯] == [1, 2, 3, 4, 5]
    assert 다섯[:2] == 앞둘


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
    from game.serving import store

    주소 = test_db_url(tmp_path, "lb_api.db")
    monkeypatch.setenv("BJ_DATABASE_URL", 주소)
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    reset_engine()
    cache.clear()
    store.reset()     # 왜: 전역 모델 저장소가 앞 테스트의 DB 내용을 들고 있을 수 있다
    엔진 = make_engine(주소)
    fresh_schema(엔진)
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
