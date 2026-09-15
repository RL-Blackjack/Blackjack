"""이 파일은 한 라운드를 어느 상태에서 시작할지 고른다.
입력: 난수 생성기
출력: StartState(지정 시작) 또는 None(자연 딜)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StartState:
    """라운드를 특정 상황에서 시작시키기 위한 지정값.

    player_cards: 플레이어에게 강제로 줄 카드(값 1~10, A=1)
    dealer_up: 딜러 업카드(값 1~10). 홀카드는 환경이 무작위로 뽑는다.
    forced_first_action: 첫 결정을 이 행동으로 강제한다(탐험적 시작·셀 검증용).
    """

    player_cards: list[int]
    dealer_up: int
    forced_first_action: int | None = None


class StartSampler(Protocol):
    """시작 상태 샘플러의 공통 모양."""

    def sample(self, rng: np.random.Generator) -> StartState | None: ...


class NaturalDeal:
    """자연 딜 — 실제 게임과 같은 분포로 시작한다.

    # 왜 None을 돌려주는가: 환경이 이미 올바른 딜 절차를 갖고 있으므로,
    #   '지정하지 않는다'를 None 하나로 표현하면 환경에 분기가 하나만 생긴다.
    """

    def sample(self, rng: np.random.Generator) -> StartState | None:
        return None
