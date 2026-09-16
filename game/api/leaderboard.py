"""이 파일은 공개 리더보드 엔드포인트를 제공한다.
입력: 개수 제한 질의 매개변수.
출력: 일치율 순으로 줄 세운 순위표.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from blackjack_rl.rules import RULES_V1
from game.db import get_session
from game.leaderboard import TOP_N, cached_leaderboard

router = APIRouter(prefix="/api", tags=["leaderboard"])


class LeaderRowOut(BaseModel):
    """순위표 한 줄. 이메일 같은 개인정보는 담지 않는다."""

    rank: int
    user_id: int
    display_name: str
    games: int
    decisions: int
    agreement: float
    ev_loss_per_decision: float
    net_result: float


@router.get("/leaderboard", response_model=list[LeaderRowOut])
def read_leaderboard(
    session: Annotated[Session, Depends(get_session)],
    # 왜 상한이 TOP_N인가: 캐시는 상위 TOP_N줄만 들고 있다. 그보다 크게 받으면
    #   캐시를 잘라 줄 수 없다.
    limit: Annotated[int, Query(ge=1, le=TOP_N)] = 20,
) -> list[LeaderRowOut]:
    """일치율 기준 순위표. 5분 동안 같은 결과를 돌려준다."""
    줄들 = cached_leaderboard(session, rules_fp=RULES_V1.fingerprint(), limit=limit)
    return [LeaderRowOut(**vars(r)) for r in 줄들]
