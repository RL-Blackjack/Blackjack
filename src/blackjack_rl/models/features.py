"""이 파일은 블랙잭 상태를 지도학습이 먹을 수 있는 숫자 벡터로 바꾼다.
입력: StateKey와 DP 해.
출력: 17차원 float32 특징 벡터, 최적 행동 레이블, (X, y, keys) 데이터셋.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from blackjack_rl.dp.exact import DPResult
from blackjack_rl.state import REACHABLE_KEYS, StateKey

# 왜 원핫(610차원)이 아닌가: 상태를 통째로 원핫으로 주면 모델의 첫 층이
#   조회 테이블이 되어 무조건 100%가 나온다. 그러면 '학습했다'가 아니라
#   '외웠다'이고 일반화를 잴 수 없다. 원 설계서 2.14가 DQN에서 내린 결정과
#   같은 인코딩을 써야 DQN과 MLP를 공정하게 비교할 수 있다.
FEATURE_DIM: int = 17

FEATURE_NAMES: tuple[str, ...] = (
    "total_norm",
    "is_soft",
    "up_2", "up_3", "up_4", "up_5", "up_6",
    "up_7", "up_8", "up_9", "up_10", "up_A",
    "can_double",
    "can_split",
    "split_depth_norm",
    "is_pair",
    "bias",
)

_MAX_TOTAL: float = 21.0
_MAX_DEPTH: float = 3.0


def encode(key: StateKey) -> np.ndarray:
    """상태 하나를 17차원 벡터로 만든다."""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    v[0] = key.total / _MAX_TOTAL
    v[1] = float(key.is_soft)
    # 딜러 업카드는 2~11이므로 인덱스 2~11에 원핫으로 놓는다.
    # 왜 원핫인가: 업카드는 크기 순서가 의미를 갖지 않는다. 딜러 2와 3은 가깝지만
    #   10과 A(11)는 숫자가 이웃인데 전략이 전혀 다르다. 순서를 지우는 편이 정직하다.
    v[2 + (key.dealer_up - 2)] = 1.0
    v[12] = float(key.can_double)
    v[13] = float(key.can_split)
    v[14] = key.split_depth / _MAX_DEPTH
    v[15] = float(key.can_split)   # is_pair: 스플릿 가능 = 페어라는 뜻
    v[16] = 1.0                    # bias
    return v


def encode_many(keys: Sequence[StateKey]) -> np.ndarray:
    """상태 여러 개를 (N, 17) 행렬로 만든다."""
    X = np.zeros((len(keys), FEATURE_DIM), dtype=np.float32)
    for i, key in enumerate(keys):
        X[i] = encode(key)
    return X


def optimal_actions(dp: DPResult, keys: Sequence[StateKey]) -> np.ndarray:
    """각 상태의 DP 최적 행동을 레이블 배열로 만든다."""
    y = np.zeros(len(keys), dtype=np.int8)
    for i, key in enumerate(keys):
        y[i] = int(np.nanargmax(dp.Q[key]))
    return y


def build_dataset(dp: DPResult) -> tuple[np.ndarray, np.ndarray, tuple[StateKey, ...]]:
    """도달 가능한 610개 상태로 (X, y, keys) 한 벌을 만든다."""
    keys = tuple(REACHABLE_KEYS)
    return encode_many(keys), optimal_actions(dp, keys), keys
