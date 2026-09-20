"""이 파일은 게임 플레이 API가 주고받는 모양을 정의한다.
입력: 없음(Pydantic 선언만).
출력: 요청·응답 모델들.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictInt


class OpponentOut(BaseModel):
    """상대로 고를 수 있는 모델 하나."""

    id: int
    name: str
    family: str
    exact_ev: float


class PendingOut(BaseModel):
    """다음 결정을 기다리는 라운드."""

    status: Literal["pending"] = "pending"
    seq: int
    total: int
    is_soft: bool
    dealer_up: int
    player_cards: list[int]
    legal: list[int]
    hand_index: int
    n_hands: int
    bet: float


class HandOut(BaseModel):
    """끝난 손 하나."""

    cards: list[int]
    total: int
    bet: float
    result: float


class FinishedOut(BaseModel):
    """정산까지 끝난 라운드."""

    status: Literal["finished"] = "finished"
    dealer_up: int
    dealer_cards: list[int]
    net: float
    hands: list[HandOut]


class Feedback(BaseModel):
    """방금 고른 행동에 대한 DP의 평가."""

    dp_optimal_action: int
    dp_ev_loss: float
    was_optimal: bool
    ai_action: int | None


class ActIn(BaseModel):
    """행동 요청. seq가 낙관적 잠금 역할을 한다."""

    # 왜 seq를 받는가: 새로고침이나 뒤로가기로 같은 요청이 두 번 갈 수 있다.
    #   서버가 가진 행동 개수와 다르면 409로 막는다.
    # 왜 StrictInt인가: 파이썬에서 True == 1 이라 bool이 히트(1)로 통과했고,
    #   "0"과 0.0도 조용히 정수가 되어 기록이 클라이언트 표기에 휘둘렸다.
    seq: StrictInt = Field(ge=0, le=60)
    action: StrictInt = Field(ge=0, le=3)


class ActOut(BaseModel):
    """행동 결과와 그에 대한 평가."""

    feedback: Feedback
    round: PendingOut | FinishedOut


class NewGameIn(BaseModel):
    """새 게임 요청."""

    # 왜 위아래를 막는가: 거대한 번호가 그대로 DB 드라이버에 닿아 500이 났다.
    #   모델 번호는 32비트 정수 키라 이 범위 밖은 어차피 있을 수 없다.
    opponent_model_id: int | None = Field(default=None, ge=1, le=2**31 - 1)


class RoundOut(BaseModel):
    """새로 연 라운드."""

    round_no: int
    round: PendingOut | FinishedOut


class GameOut(BaseModel):
    """게임 하나의 지금 모습."""

    game_id: int
    opponent: OpponentOut | None
    round_no: int
    round: PendingOut | FinishedOut
