"""이 파일은 DP 최적해를 한 번 풀어 두고 정답과 EV 손실을 빠르게 돌려준다.
입력: 상태 키와 고른 행동.
출력: 최적 행동 번호와 최적 대비 기대값 손실.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from blackjack_rl.dp.exact import DPResult, solve_optimal
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import STAND, StateKey


@lru_cache(maxsize=1)
def dp() -> DPResult:
    """규칙 하나에 대한 DP 최적해. 서버가 살아 있는 동안 한 번만 푼다."""
    # 왜 캐시인가: action_values()는 부를 때마다 솔버를 새로 만들어 0.2ms가 든다.
    #   solve_optimal()은 0.03초에 끝나고, 그 뒤로는 표 조회가 0.22us다(900배).
    return solve_optimal(RULES_V1)


def _q(key: StateKey) -> np.ndarray | None:
    """상태 하나의 행동가치 4개. DP가 열거하지 않은 자리면 None."""
    q = dp().Q[key.total, key.is_soft, key.dealer_up,
               key.can_double, key.can_split, key.split_depth]
    if np.all(np.isnan(q)):
        # 왜 생길 수 있는가: 무작위 4,000판에서는 한 번도 안 나왔지만, 규칙을
        #   바꾸면 DP가 안 푸는 자리가 생길 수 있다. 서버가 죽으면 안 된다.
        return None
    return q


def optimal_action(key: StateKey, legal: np.ndarray) -> int:
    """이 상태의 DP 정답. 모르는 자리면 언제나 합법인 STAND로 둔다."""
    q = _q(key)
    if q is None:
        return STAND
    가려낸 = np.where(np.asarray(legal, dtype=bool), q, np.nan)
    if np.all(np.isnan(가려낸)):
        return STAND
    return int(np.nanargmax(가려낸))


def ev_loss(key: StateKey, action: int) -> float:
    """최적 대비 기대값 손실. 최적을 골랐으면 정확히 0이고, 언제나 0 이상이다."""
    q = _q(key)
    if q is None or np.isnan(q[action]):
        return 0.0
    최선 = float(np.nanmax(q))
    # 왜 max(0.0, ...)인가: 부동소수 오차로 -1e-17 같은 값이 나올 수 있다.
    #   EV 손실이 음수면 "DP보다 잘 뒀다"는 뜻이 되어 정의가 깨진다.
    return max(0.0, 최선 - float(q[action]))
