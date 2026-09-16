"""이 파일은 일치율 기준 리더보드를 집계하고 잠시 캐시한다.
입력: DB 세션과 규칙 지문.
출력: 순위가 매겨진 LeaderRow 목록.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Float, case, cast, func, select
from sqlalchemy.orm import Session

from game.models import Decision, Game, Round, User
from game.stats import MIN_DECISIONS_FOR_RANK

# 왜 5분인가: 설계서 §6.3이 정했다. 머티리얼라이즈드 뷰를 못 쓰게 됐으므로
#   같은 효과를 애플리케이션 계층에서 낸다. 리더보드가 5분 늦어도 문제가 없다.
CACHE_SECONDS: float = 300.0
# 왜 100인가: API가 허용하는 limit 상한이다. 이만큼 한 번 세어 두면 어떤 limit도
#   잘라서 줄 수 있다.
TOP_N: int = 100


@dataclass(frozen=True)
class LeaderRow:
    """리더보드 한 줄. 개인정보는 표시명 말고 아무것도 담지 않는다."""

    rank: int
    user_id: int
    display_name: str
    games: int
    decisions: int
    agreement: float
    ev_loss_per_decision: float
    net_result: float


class TTLCache:
    """값을 정해진 시간만큼 들고 있는 아주 작은 캐시. 여러 스레드가 같이 써도 된다."""

    def __init__(self, ttl_seconds: float = CACHE_SECONDS) -> None:
        self.ttl = ttl_seconds
        self._칸: dict[str, tuple[float, Any]] = {}
        # 왜 잠금인가: 동기 핸들러는 스레드 풀에서 돌고 이 캐시는 모듈 전역 하나다.
        #   읽기와 만료 삭제 사이에 다른 스레드가 끼면 KeyError가 나거나(실측 재현),
        #   끼어든 스레드가 막 넣은 새 값을 지운다. 세 동작 모두 한 잠금 안에서 한다.
        self._잠금 = threading.Lock()

    def get(self, key: str, *, now: float | None = None) -> Any | None:
        """아직 안 상한 값이면 준다. 없거나 상했으면 None."""
        시각 = now if now is not None else time.monotonic()
        with self._잠금:
            담긴것 = self._칸.get(key)
            if 담긴것 is None:
                return None
            넣은때, 값 = 담긴것
            if 시각 - 넣은때 > self.ttl:
                # 왜 del이 아니라 pop인가: 잠금을 바꾸거나 빼도 이미 지워진 키에서
                #   요청이 500으로 죽지 않게 하는 두 번째 방어선이다.
                self._칸.pop(key, None)
                return None
            return 값

    def put(self, key: str, value: Any, *, now: float | None = None) -> None:
        """값을 넣는다."""
        시각 = now if now is not None else time.monotonic()
        with self._잠금:
            self._칸[key] = (시각, value)

    def clear(self) -> None:
        """전부 지운다."""
        with self._잠금:
            self._칸.clear()


cache = TTLCache()


def cached_leaderboard(session: Session, *, rules_fp: str, limit: int = 20,
                       ttl_cache: TTLCache | None = None) -> list[LeaderRow]:
    """상위 TOP_N줄을 규칙 지문마다 한 번 세어 캐시하고 limit만큼 잘라 준다."""
    캐시 = ttl_cache if ttl_cache is not None else cache
    # 왜 키에 limit이 없는가: 인증 없는 공개 API라 limit=1~100을 바꿔 가며 부르면
    #   5분마다 전체 집계를 100번 일으킬 수 있었다. 키는 규칙 지문 하나다.
    줄들 = 캐시.get(rules_fp)
    if 줄들 is None:
        # 왜 튜플인가: 여러 요청이 같은 캐시 값을 나눠 쓴다. 바꿀 수 없게 담는다.
        줄들 = tuple(leaderboard(session, rules_fp=rules_fp, limit=TOP_N))
        캐시.put(rules_fp, 줄들)
    return list(줄들[:limit])


def leaderboard(session: Session, *, rules_fp: str,
                limit: int = 20) -> list[LeaderRow]:
    """일치율이 높은 순으로 줄을 세운다. 최소 결정 수를 못 채우면 뺀다."""
    맞음 = cast(case((Decision.action_taken == Decision.dp_optimal_action, 1),
                    else_=0), Float)

    # 왜 결정 집계를 따로 묶는가: 게임 합계와 한 조인에 넣으면 게임 순손익이
    #   결정 수만큼 부풀어, 예전엔 사람마다 쿼리를 한 번 더 쳤다(1+N번).
    #   사용자별로 한 줄씩 만든 두 집계를 붙이면 사용자 수와 무관하게 한 번이다.
    # 왜 라운드가 열렸는지 거르지 않는가: 결정은 내린 순간 확정된 사실이다. 빼면
    #   틀린 뒤 라운드를 버려 일치율을 지울 수 있다(/api/me/stats와 같은 정의).
    결정 = (
        select(
            Game.user_id.label("user_id"),
            func.count(Decision.id).label("decisions"),
            func.avg(맞음).label("agreement"),
            func.sum(Decision.dp_ev_loss).label("ev_loss"),
        )
        .select_from(Game)
        .join(Round, Round.game_id == Game.id)
        .join(Decision, Decision.round_id == Round.id)
        .where(Game.rules_fp == rules_fp)
        .group_by(Game.user_id)
        # 왜 HAVING인가: 설계서 §5.3의 "최소 200결정" 자격이다. 표본이 적으면
        #   일치율도 흔들려 운 좋은 사람이 1등을 한다.
        .having(func.count(Decision.id) >= MIN_DECISIONS_FOR_RANK)
        .subquery("decision_agg")
    )
    # 왜 결정이 없는 게임도 세는가: games와 net_result는 같은 게임 집합(규칙 지문이
    #   같은 게임 전체)을 가리켜야 한다. 결정이 있는 게임만 세면 결정 없이 버린
    #   게임이 게임 수에서는 숨고 순손익에는 남는다.
    게임 = (
        select(
            Game.user_id.label("user_id"),
            func.count(Game.id).label("games"),
            func.sum(Game.net_result).label("net"),
        )
        .where(Game.rules_fp == rules_fp)
        .group_by(Game.user_id)
        .subquery("game_agg")
    )

    질의 = (
        select(User.id, User.display_name, 게임.c.games, 결정.c.decisions,
               결정.c.agreement, 결정.c.ev_loss, 게임.c.net)
        .select_from(결정)
        .join(User, User.id == 결정.c.user_id)
        .join(게임, 게임.c.user_id == 결정.c.user_id)
        # 왜 일치율 순인가: 승률은 운이 지배한다. 실력으로 줄을 세워야 한다.
        .order_by(결정.c.agreement.desc(), 결정.c.decisions.desc(), User.id)
        .limit(limit)
    )

    return [
        LeaderRow(
            rank=순위, user_id=int(uid), display_name=이름,
            games=int(게임수), decisions=int(결정수),
            agreement=float(일치 or 0.0),
            ev_loss_per_decision=(float(손실 or 0.0) / 결정수) if 결정수 else 0.0,
            net_result=float(순손익 or 0.0))
        for 순위, (uid, 이름, 게임수, 결정수, 일치, 손실, 순손익) in enumerate(
            session.execute(질의).all(), start=1)
    ]
