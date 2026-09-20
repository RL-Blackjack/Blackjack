"""이 파일은 DP 최적해를 한 번 풀어 두고 정답과 EV 손실을 빠르게 돌려준다.
입력: 상태 키, 고른 행동, 그리고 손의 실제 스플릿 깊이.
출력: 최적 행동 번호와 최적 대비 기대값 손실.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from blackjack_rl.dp.exact import DPResult, solve_optimal
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import Q_SHAPE, STAND, StateKey

# 왜 표에서 끌어오는가: DP 배열의 깊이 축 길이가 곧 쓸 수 있는 깊이 수다.
#   숫자를 따로 적어 두면 어블레이션으로 축이 늘어날 때 한 곳이 어긋난다.
MAX_SPLIT_DEPTH: int = Q_SHAPE[5] - 1


@lru_cache(maxsize=1)
def dp() -> DPResult:
    """규칙 하나에 대한 DP 최적해. 서버가 살아 있는 동안 한 번만 푼다."""
    # 왜 캐시인가: action_values()는 부를 때마다 솔버를 새로 만들어 0.2ms가 든다.
    #   solve_optimal()은 0.03초에 끝나고, 그 뒤로는 표 조회가 0.22us다(900배).
    return solve_optimal(RULES_V1)


def _깊이키(key: StateKey, depth: int | None) -> StateKey:
    """조회에 쓸 키. 실제 깊이를 받으면 그 깊이의 칸을 보게 바꾼다."""
    # 왜 필요한가: RULES_V1은 키에서 스플릿 깊이를 빼 key.split_depth가 늘 0이다.
    #   깊이 2에서 갈라진 손을 깊이 0 Q로 재면 손실이 최대 0.111 과대로 나왔다
    #   (무작위 5,002결정 중 2건). 엔진이 알려 준 실제 깊이로 조회해야 맞다.
    if depth is None:
        return key
    if not 0 <= int(depth) <= MAX_SPLIT_DEPTH:
        # 왜 예외인가: 조용히 자르면 엉뚱한 칸의 Q로 손실을 재고도 아무도 모른다.
        raise ValueError(f"스플릿 깊이가 0~{MAX_SPLIT_DEPTH} 밖이다: {depth!r}")
    return key._replace(split_depth=int(depth))


def _q(key: StateKey) -> np.ndarray | None:
    """상태 하나의 행동가치 4개. DP가 열거하지 않은 자리면 None."""
    q = dp().Q[key.total, key.is_soft, key.dealer_up,
               key.can_double, key.can_split, key.split_depth]
    if np.all(np.isnan(q)):
        # 왜 생길 수 있는가: 무작위 4,000판에서는 한 번도 안 나왔지만, 규칙을
        #   바꾸면 DP가 안 푸는 자리가 생길 수 있다. 서버가 죽으면 안 된다.
        return None
    return q


def optimal_action(key: StateKey, legal: np.ndarray, *,
                   depth: int | None = None) -> int:
    """이 상태의 DP 정답. 모르는 자리면 언제나 합법인 STAND로 둔다."""
    q = _q(_깊이키(key, depth))
    if q is None:
        return STAND
    가려낸 = np.where(np.asarray(legal, dtype=bool), q, np.nan)
    if np.all(np.isnan(가려낸)):
        return STAND
    return int(np.nanargmax(가려낸))


def ev_loss(key: StateKey, action: int, *, depth: int | None = None) -> float:
    """최적 대비 기대값 손실. 최적을 골랐으면 정확히 0이고, 언제나 0 이상이다."""
    q = _q(_깊이키(key, depth))
    if q is None or np.isnan(q[action]):
        return 0.0
    최선 = float(np.nanmax(q))
    # 왜 max(0.0, ...)인가: 부동소수 오차로 -1e-17 같은 값이 나올 수 있다.
    #   EV 손실이 음수면 "DP보다 잘 뒀다"는 뜻이 되어 정의가 깨진다.
    return max(0.0, 최선 - float(q[action]))
