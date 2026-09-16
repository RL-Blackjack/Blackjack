"""이 파일은 한 사람의 전적을 집계한다.
입력: DB 세션과 사용자 번호.
출력: 승률·일치율·EV 손실을 담은 UserStats와 최근 게임 목록.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Float, and_, case, cast, func, select
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
    # 왜 두 집계의 범위가 다른가: 라운드 수·승패무는 닫힌 라운드만, 결정은 기록된
    #   것 전부를 센다. 진행 중인 라운드는 net이 아직 0.0이라 넣으면 무승부로 세어진다.
    #   결정의 정답·EV 손실은 결정 시점에 확정된 사실이고, 열린 라운드의 결정을 빼면
    #   오답을 낸 뒤 라운드를 버려 일치율에서 지울 수 있다.
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
        # 왜 WHERE가 아니라 ON 절인가: WHERE에 두면 열린 라운드만 있는 게임은 조인
        #   행이 통째로 빠져 게임 수에서도 사라진다(실측 2→1). ON에 두면 그 게임은
        #   라운드 없는 행 하나로 남아 게임 수에만 든다.
        .outerjoin(Round, and_(Round.game_id == Game.id, Round.is_open.is_(False)))
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


def recent_games(session: Session, user_id: int, *, rules_fp: str,
                 limit: int = 10) -> list[GameSummary]:
    """최근 게임을 새것부터 준다. 규칙 지문이 같은 게임만 준다."""
    # 왜 규칙으로 거르는가: 규칙이 바뀐 옛 게임은 열면 409(rules_changed)라 이어 둘 수
    #   없고, 일치율도 다른 DP 정답 기준이다. user_stats와 같은 게임 집합을 보여 준다.
    # 왜 두 서브쿼리를 이 사람의 게임으로 좁히는가: 안 거르면 모든 사람의 라운드·결정을
    #   게임별로 묶은 뒤 바깥 조인에서 버린다. 결과는 같지만 사용자가 늘수록 내 목록
    #   조회가 느려진다(실측: 남의 결정 500개에 SQLite 명령 수 253→10,043).
    내게임 = select(Game.id).where(Game.user_id == user_id, Game.rules_fp == rules_fp)
    # 라운드 수는 user_stats처럼 닫힌 라운드만, 일치율은 기록된 결정 전부로 센다.
    라운드수 = (
        select(Round.game_id, func.count(Round.id).label("n"))
        .where(Round.game_id.in_(내게임), Round.is_open.is_(False))
        .group_by(Round.game_id).subquery())
    일치 = (
        select(Round.game_id.label("gid"),
               func.count(Decision.id).label("d"),
               func.coalesce(func.sum(_맞음()), 0.0).label("ok"))
        .join(Decision, Decision.round_id == Round.id)
        .where(Round.game_id.in_(내게임))
        .group_by(Round.game_id).subquery())

    행들 = session.execute(
        select(Game.id, ModelRegistry.name, Game.started_at, Game.ended_at,
               func.coalesce(라운드수.c.n, 0), Game.net_result,
               func.coalesce(일치.c.d, 0), func.coalesce(일치.c.ok, 0.0))
        .outerjoin(ModelRegistry, ModelRegistry.id == Game.opponent_model_id)
        .outerjoin(라운드수, 라운드수.c.game_id == Game.id)
        .outerjoin(일치, 일치.c.gid == Game.id)
        .where(Game.user_id == user_id, Game.rules_fp == rules_fp)
        .order_by(Game.started_at.desc(), Game.id.desc())
        .limit(limit)
    ).all()

    return [
        GameSummary(game_id=int(gid), opponent_name=이름, started_at=시작,
                    ended_at=끝, rounds=int(n), net_result=float(순손익),
                    agreement=(float(ok) / d) if d else 0.0)
        for gid, 이름, 시작, 끝, n, 순손익, d, ok in 행들
    ]
