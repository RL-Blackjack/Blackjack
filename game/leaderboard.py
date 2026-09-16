"""이 파일은 일치율 기준 리더보드를 집계하고 잠시 캐시한다.
입력: DB 세션과 규칙 지문.
출력: 순위가 매겨진 LeaderRow 목록.
"""

from __future__ import annotations

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
    """값을 정해진 시간만큼 들고 있는 아주 작은 캐시."""

    def __init__(self, ttl_seconds: float = CACHE_SECONDS) -> None:
        self.ttl = ttl_seconds
        self._칸: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, *, now: float | None = None) -> Any | None:
        """아직 안 상한 값이면 준다. 없거나 상했으면 None."""
        시각 = now if now is not None else time.monotonic()
        담긴것 = self._칸.get(key)
        if 담긴것 is None:
            return None
        넣은때, 값 = 담긴것
        if 시각 - 넣은때 > self.ttl:
            del self._칸[key]
            return None
        return 값

    def put(self, key: str, value: Any, *, now: float | None = None) -> None:
        """값을 넣는다."""
        self._칸[key] = (now if now is not None else time.monotonic(), value)

    def clear(self) -> None:
        """전부 지운다."""
        self._칸.clear()


cache = TTLCache()


def leaderboard(session: Session, *, rules_fp: str,
                limit: int = 20) -> list[LeaderRow]:
    """일치율이 높은 순으로 줄을 세운다. 최소 결정 수를 못 채우면 뺀다."""
    맞음 = cast(case((Decision.action_taken == Decision.dp_optimal_action, 1),
                    else_=0), Float)

    질의 = (
        select(
            User.id,
            User.display_name,
            func.count(func.distinct(Game.id)).label("games"),
            func.count(Decision.id).label("decisions"),
            func.avg(맞음).label("agreement"),
            func.sum(Decision.dp_ev_loss).label("ev_loss"),
            func.sum(Game.net_result).label("net"),
        )
        .select_from(User)
        .join(Game, Game.user_id == User.id)
        .join(Round, Round.game_id == Game.id)
        .join(Decision, Decision.round_id == Round.id)
        .where(Game.rules_fp == rules_fp)
        .group_by(User.id, User.display_name)
        # 왜 HAVING인가: 설계서 §5.3의 "최소 200결정" 자격이다. 표본이 적으면
        #   일치율도 흔들려 운 좋은 사람이 1등을 한다.
        .having(func.count(Decision.id) >= MIN_DECISIONS_FOR_RANK)
        # 왜 일치율 순인가: 승률은 운이 지배한다. 실력으로 줄을 세워야 한다.
        .order_by(func.avg(맞음).desc(), func.count(Decision.id).desc(), User.id)
        .limit(limit)
    )

    줄들 = []
    for 순위, (uid, 이름, 게임수, 결정수, 일치, 손실, 순손익) in enumerate(
            session.execute(질의).all(), start=1):
        # 왜 게임 순손익을 결정 수로 나누지 않는가: net은 게임 단위 합계이고
        #   조인 때문에 결정 수만큼 부풀어 있다. 그래서 별도로 다시 센다.
        실제순손익 = session.scalar(
            select(func.coalesce(func.sum(Game.net_result), 0.0))
            .where(Game.user_id == uid, Game.rules_fp == rules_fp)) or 0.0
        줄들.append(LeaderRow(
            rank=순위, user_id=int(uid), display_name=이름,
            games=int(게임수), decisions=int(결정수),
            agreement=float(일치 or 0.0),
            ev_loss_per_decision=(float(손실 or 0.0) / 결정수) if 결정수 else 0.0,
            net_result=float(실제순손익)))
    return 줄들
