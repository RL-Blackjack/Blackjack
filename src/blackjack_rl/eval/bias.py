"""이 파일은 학습 중 Q 추정치가 DP 최적가치보다 얼마나 위로 떠 있는지 잰다.
입력: 에이전트의 Q 테이블(Q_SHAPE)과 DP가 준 최적 상태가치 V(Q_SHAPE[:-1]).
출력: max_a Q(s,a) - V*(s) 를 프로브 상태들에 대해 평균한 float 하나.
"""

from __future__ import annotations

import numpy as np

from blackjack_rl.state import LEGAL, REACHABLE_KEYS, StateKey


def _make_probe_keys() -> tuple[StateKey, ...]:
    """편향을 재기에 가장 적당한 상태들을 고른다."""
    keys = []
    for key in REACHABLE_KEYS:
        if key.is_soft != 0:
            continue
        if key.can_double != 0 or key.can_split != 0:
            continue
        if not 12 <= key.total <= 16:
            continue
        keys.append(key)
    return tuple(keys)


# 왜 하필 하드 12~16(한 번 이상 히트한 뒤)인가:
#   1) 가장 많이 방문되는 구간이다. 100만판 학습에서 칸당 방문수 중앙값이 2,596회로
#      (최소 1,277, 최대 16,462), "아직 안 배워서 Q가 0에 가깝다"는 잡음이 최소가 된다.
#   2) STAND와 HIT의 값이 서로 가깝다. 두 추정치가 가까울수록 max 연산이
#      잡음을 위로 퍼올리는 효과(최대화 편향)가 크게 드러난다.
#   3) 딜러 업카드 10칸을 모두 넣어 특정 칸을 유리하게 고르지 않았음을 보인다.
#   결과적으로 50칸이고, 이 칸들의 DP 최적가치 평균은 -0.3079509031이다.
#   (610칸 전체로 재면 V* 평균이 +0.0906이라 완전히 다른 숫자가 나온다.
#    발표의 '편향 곡선'은 반드시 이 50칸으로 잰 값이어야 한다.)
PROBE_KEYS: tuple[StateKey, ...] = _make_probe_keys()


def probe_vstar_mean(v_star: np.ndarray) -> float:
    """프로브 칸들의 DP 최적 상태가치 평균. 프로브 정의가 바뀌면 이 값이 흔들린다."""
    return float(np.mean([float(v_star[key]) for key in PROBE_KEYS]))


def maxq_minus_vstar(
    q_table: np.ndarray,
    v_star: np.ndarray,
    keys: tuple[StateKey, ...] = PROBE_KEYS,
) -> float:
    """max_a Q(s,a) - V*(s) 를 프로브 상태에 대해 평균한다.

    양수면 에이전트가 자기 실력을 과대평가하고 있다는 뜻이다.
    Q러닝은 max 연산이 추정 잡음을 위로 퍼올려 이 값이 위쪽에 서고,
    Double Q는 고르는 표와 읽는 표를 분리해 그 연결고리를 끊는다.
    doubleq 에이전트는 q_table에 q_eval_table()((Q+QB)/2)을 넣어 부른다.
    """
    차이들 = []
    for key in keys:
        # 왜 where로 마스크하는가: 불법 행동의 Q는 -inf다. 그대로 max에 넣으면
        #     상관없지만, Q 초기화를 nan으로 바꾸는 날 조용히 nan이 전파된다.
        best = float(np.max(np.where(LEGAL[key], q_table[key], -np.inf)))
        차이들.append(best - float(v_star[key]))
    return float(np.mean(차이들))
