"""이 파일은 상태를 특징 벡터로 바꾸는 인코딩이 정확하고 유일한지 확인한다.
입력: REACHABLE_KEYS 610개와 DP 해.
출력: 차원·범위·유일성·레이블 분포에 대한 pytest 결과.
"""

import numpy as np
import pytest

from blackjack_rl.models.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    build_dataset,
    encode,
    encode_many,
    optimal_actions,
)
from blackjack_rl.state import DOUBLE, HIT, REACHABLE_KEYS, SPLIT, STAND, StateKey


def test_차원이_17이고_이름이_그만큼_있다():
    assert FEATURE_DIM == 17
    assert len(FEATURE_NAMES) == 17
    assert len(set(FEATURE_NAMES)) == 17


def test_한_상태를_인코딩하면_17차원_float32다():
    v = encode(StateKey(16, 0, 10, 1, 0, 0))
    assert v.shape == (17,)
    assert v.dtype == np.float32


def test_모든_값이_0과_1_사이다():
    # 왜: 정규화가 깨지면 MLP가 학습을 못 한다. 610개 전부 확인한다.
    X = encode_many(REACHABLE_KEYS)
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_딜러_업카드가_원핫이다():
    v = encode(StateKey(16, 0, 10, 1, 0, 0))
    원핫 = v[2:12]
    assert 원핫.sum() == pytest.approx(1.0)
    assert 원핫[10 - 2] == pytest.approx(1.0)

    # 왜 v11[11]인가: 위의 '원핫'은 v[2:12] 슬라이스라 상대 인덱스(0~9)를 쓰지만,
    #   여기서는 17차원 원본 벡터를 직접 본다. dealer_up=11은 v[2+(11-2)] = v[11]이다.
    #   슬라이스 인덱스와 원본 인덱스를 섞어 쓰면 엉뚱한 칸을 검사하게 된다.
    v11 = encode(StateKey(16, 0, 11, 1, 0, 0))
    assert v11[11] == pytest.approx(1.0)
    assert v11[2:12].sum() == pytest.approx(1.0)


def test_610개_상태가_서로_다른_벡터가_된다():
    # 왜: 두 상태가 같은 벡터가 되면 모델이 원리적으로 구분할 수 없다.
    #     인코딩이 정보를 잃지 않았다는 가장 중요한 검사다.
    X = encode_many(REACHABLE_KEYS)
    assert X.shape == (610, 17)
    유일 = np.unique(X, axis=0)
    assert len(유일) == 610


def test_encode_many가_encode를_쌓은_것과_같다():
    키들 = REACHABLE_KEYS[:20]
    쌓기 = np.stack([encode(k) for k in 키들])
    assert np.array_equal(encode_many(키들), 쌓기)


def test_레이블은_네_행동_중_하나다(dp):
    y = optimal_actions(dp, REACHABLE_KEYS)
    assert y.shape == (610,)
    assert y.dtype == np.int8
    assert set(np.unique(y).tolist()) <= {STAND, HIT, DOUBLE, SPLIT}


def test_레이블_분포가_실측값과_같다(dp):
    # 왜 이 숫자인가: 2026-09-16에 직접 세어 본 값이다. 이 분포가 흔들리면
    #     DP나 도달 가능 상태 정의가 바뀐 것이므로 즉시 알아야 한다.
    y = optimal_actions(dp, REACHABLE_KEYS)
    센값 = {int(a): int((y == a).sum()) for a in np.unique(y)}
    assert 센값 == {STAND: 219, HIT: 294, DOUBLE: 45, SPLIT: 52}


def test_빌드된_데이터셋의_모양이_맞는다(dp):
    X, y, keys = build_dataset(dp)
    assert X.shape == (610, 17)
    assert y.shape == (610,)
    assert len(keys) == 610
    assert tuple(keys) == tuple(REACHABLE_KEYS)


def test_명백한_칸의_레이블이_상식과_맞는다(dp):
    X, y, keys = build_dataset(dp)
    자리 = {k: i for i, k in enumerate(keys)}
    assert y[자리[StateKey(20, 0, 6, 1, 0, 0)]] == STAND
    assert y[자리[StateKey(5, 0, 10, 1, 0, 0)]] == HIT
    assert y[자리[StateKey(11, 0, 6, 1, 0, 0)]] == DOUBLE
