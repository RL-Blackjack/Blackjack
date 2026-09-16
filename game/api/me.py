"""이 파일은 로그인한 사람의 전적과 최근 게임을 돌려준다.
입력: 액세스 토큰.
출력: 승률·일치율·EV 손실과 게임 목록.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from blackjack_rl.rules import RULES_V1
from game.db import get_session
from game.deps import current_user
from game.models import User
from game.stats import recent_games, user_stats

router = APIRouter(prefix="/api/me", tags=["me"])


class StatsOut(BaseModel):
    """전적 한 장. 설계서 §5.3의 세 지표를 함께 보여 준다."""

    games: int
    rounds: int
    decisions: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    agreement: float
    ev_loss_total: float
    ev_loss_per_decision: float
    net_result: float
    rank_eligible: bool


class GameRowOut(BaseModel):
    """게임 목록의 한 줄."""

    game_id: int
    opponent_name: str | None
    started_at: datetime
    ended_at: datetime | None
    rounds: int
    net_result: float
    agreement: float


@router.get("/stats", response_model=StatsOut)
def my_stats(사용자: Annotated[User, Depends(current_user)],
             session: Annotated[Session, Depends(get_session)]) -> StatsOut:
    """내 전적. 지금 규칙으로 둔 게임만 센다."""
    통계 = user_stats(session, 사용자.id, rules_fp=RULES_V1.fingerprint())
    return StatsOut(**vars(통계))


@router.get("/games", response_model=list[GameRowOut])
def my_games(사용자: Annotated[User, Depends(current_user)],
             session: Annotated[Session, Depends(get_session)],
             limit: Annotated[int, Query(ge=1, le=100)] = 10) -> list[GameRowOut]:
    """최근 게임 목록. 지금 규칙으로 둔 게임만 준다."""
    # 왜 전적과 같은 지문으로 거르는가: 규칙이 바뀐 옛 게임은 열면 409라 이어 둘 수
    #   없다. 목록에 두면 누를 수 없는 줄이 되고, 전적 숫자와도 어긋난다.
    return [GameRowOut(**vars(g))
            for g in recent_games(session, 사용자.id,
                                  rules_fp=RULES_V1.fingerprint(), limit=limit)]
