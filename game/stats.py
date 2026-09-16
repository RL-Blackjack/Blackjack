"""이 파일은 한 사람의 전적을 집계한다.
입력: DB 세션과 사용자 번호.
출력: 승률·일치율·EV 손실을 담은 UserStats와 최근 게임 목록.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Float, case, cast, func, select
from sqlalchemy.orm import Session

from game.models import Decision, Game, ModelRegistry, Round

# 왜 200인가: 설계서 §5.3이 정했다. 블랙잭은 핸드당 표준편차가 1.15단위라
#   표본이 적으면 일치율도 흔들린다.
MIN_DECISIONS_FOR_RANK: int = 200


@dataclass(frozen=True)
class UserStats:
    """한 사람의 전적 한 장."""

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


@dataclass(frozen=True)
class GameSummary:
    """게임 목록에 한 줄로 보여 줄 요약."""

    game_id: int
    opponent_name: str | None
    started_at: datetime
    ended_at: datetime | None
    rounds: int
    net_result: float
    agreement: float


def _맞음():
    """행동이 DP 정답과 같으면 1.0, 아니면 0.0인 식."""
    # 왜 cast(Float)인가: SQLite와 PostgreSQL 모두에서 AVG가 정수 나눗셈으로
    #   0이 되지 않게 하려면 실수로 올려 줘야 한다.
    return cast(case((Decision.action_taken == Decision.dp_optimal_action, 1),
                     else_=0), Float)


def user_stats(session: Session, user_id: int, *, rules_fp: str) -> UserStats:
    """한 사람의 전적을 집계한다. 규칙 지문이 같은 게임만 센다."""
    라운드집계 = session.execute(
        select(
            func.count(func.distinct(Game.id)),
            func.count(Round.id),
            func.coalesce(func.sum(case((Round.net > 0, 1), else_=0)), 0),
            func.coalesce(func.sum(case((Round.net < 0, 1), else_=0)), 0),
            func.coalesce(func.sum(case((Round.net == 0, 1), else_=0)), 0),
            func.coalesce(func.sum(Round.net), 0.0),
        )
        .select_from(Game)
        .outerjoin(Round, Round.game_id == Game.id)
        .where(Game.user_id == user_id, Game.rules_fp == rules_fp)
    ).one()
    게임수, 라운드수, 승, 패, 무, 순손익 = 라운드집계

    결정집계 = session.execute(
        select(
            func.count(Decision.id),
            func.coalesce(func.sum(_맞음()), 0.0),
            func.coalesce(func.sum(Decision.dp_ev_loss), 0.0),
        )
        .select_from(Game)
        .join(Round, Round.game_id == Game.id)
        .join(Decision, Decision.round_id == Round.id)
        .where(Game.user_id == user_id, Game.rules_fp == rules_fp)
    ).one()
    결정수, 맞은수, 손실합 = 결정집계

    return UserStats(
        games=int(게임수), rounds=int(라운드수), decisions=int(결정수),
        wins=int(승), losses=int(패), pushes=int(무),
        # 왜 이렇게 나누는가: 방금 가입한 사람은 분모가 0이다. 그때 터지면
        #   첫 화면이 안 뜬다.
        win_rate=(float(승) / 라운드수) if 라운드수 else 0.0,
        agreement=(float(맞은수) / 결정수) if 결정수 else 0.0,
        ev_loss_total=float(손실합),
        ev_loss_per_decision=(float(손실합) / 결정수) if 결정수 else 0.0,
        net_result=float(순손익),
        rank_eligible=int(결정수) >= MIN_DECISIONS_FOR_RANK,
    )


def recent_games(session: Session, user_id: int, *,
                 limit: int = 10) -> list[GameSummary]:
    """최근 게임을 새것부터 준다."""
    라운드수 = (
        select(Round.game_id, func.count(Round.id).label("n"))
        .group_by(Round.game_id).subquery())
    일치 = (
        select(Round.game_id.label("gid"),
               func.count(Decision.id).label("d"),
               func.coalesce(func.sum(_맞음()), 0.0).label("ok"))
        .join(Decision, Decision.round_id == Round.id)
        .group_by(Round.game_id).subquery())

    행들 = session.execute(
        select(Game.id, ModelRegistry.name, Game.started_at, Game.ended_at,
               func.coalesce(라운드수.c.n, 0), Game.net_result,
               func.coalesce(일치.c.d, 0), func.coalesce(일치.c.ok, 0.0))
        .outerjoin(ModelRegistry, ModelRegistry.id == Game.opponent_model_id)
        .outerjoin(라운드수, 라운드수.c.game_id == Game.id)
        .outerjoin(일치, 일치.c.gid == Game.id)
        .where(Game.user_id == user_id)
        .order_by(Game.started_at.desc(), Game.id.desc())
        .limit(limit)
    ).all()

    return [
        GameSummary(game_id=int(gid), opponent_name=이름, started_at=시작,
                    ended_at=끝, rounds=int(n), net_result=float(순손익),
                    agreement=(float(ok) / d) if d else 0.0)
        for gid, 이름, 시작, 끝, n, 순손익, d, ok in 행들
    ]
