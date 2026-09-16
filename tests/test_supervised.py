"""이 파일은 지도학습 두 모델이 제대로 학습되고 정책을 내놓는지 확인한다.
입력: DP로 만든 (X, y) 데이터셋과 분할.
출력: 정확도·정책 형태·재현성에 대한 pytest 결과.
"""

import numpy as np
import pytest

from blackjack_rl.models.features import build_dataset
from blackjack_rl.models.splits import make_split
from blackjack_rl.models.supervised import MODEL_KINDS, make_estimator, train


@pytest.fixture(scope="module")
def 데이터(dp):
    return build_dataset(dp)


def test_모델_종류가_두_가지다():
    assert MODEL_KINDS == ("rf", "mlp")


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_추정기를_만들_수_있다(kind):
    est = make_estimator(kind, seed=0)
    assert hasattr(est, "fit") and hasattr(est, "predict")


# 왜 모델마다 기준이 다른가: 610칸을 전부 줬을 때 랜덤포레스트는 1.000으로 완전히
#   외우지만 MLP는 0.937(5시드 0.925~0.943)에서 멈춘다. 처음에는 122칸·366칸 결과가
#   둘 다 1.000이라 610칸도 그럴 것으로 추정했으나, 실제로 재 보니 틀렸다.
#   **이것은 버그가 아니라 발견이다** — MLP는 610칸을 외울 용량이 없고, 그래서
#   정답을 100% 줘도 EV가 -1.584%에 머문다(랜덤포레스트는 -0.511%).
TRAIN_ACC_FLOOR: dict[str, float] = {"rf": 0.99, "mlp": 0.90}


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_전부_학습하면_훈련정확도가_모델별_기준을_넘는다(데이터, kind):
    X, y, keys = 데이터
    m = train(kind, X, y, make_split("random", keys, 1.0, seed=0), seed=0)
    assert m.train_acc >= TRAIN_ACC_FLOOR[kind]


def test_랜덤포레스트는_외우고_MLP는_못_외운다(데이터):
    """정답을 전부 줬을 때 두 모델의 용량 차이를 수치로 못박는다.

    이 차이가 EV 비교의 근거다. 랜덤포레스트가 DP와 똑같은 -0.511%를 내는 것은
    학습이 아니라 표를 복사했기 때문이고, MLP가 -1.584%에 머무는 것은 애초에
    610칸을 담을 용량이 없기 때문이다.
    """
    X, y, keys = 데이터
    전부 = make_split("random", keys, 1.0, seed=0)
    rf = train("rf", X, y, 전부, seed=0)
    mlp = train("mlp", X, y, 전부, seed=0)
    assert rf.train_acc == pytest.approx(1.0, abs=1e-9)
    assert 0.90 <= mlp.train_acc < 0.99
    assert rf.train_acc > mlp.train_acc


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_정책은_610칸_int8이고_전부_합법_행동이다(데이터, kind):
    X, y, keys = 데이터
    m = train(kind, X, y, make_split("random", keys, 0.6, seed=0), seed=0)
    pol = m.policy_full(X)
    assert pol.shape == (610,)
    assert pol.dtype == np.int8
    assert set(np.unique(pol).tolist()) <= {0, 1, 2, 3}


def test_같은_시드면_같은_정책이_나온다(데이터):
    X, y, keys = 데이터
    s = make_split("structural", keys, 0.6, seed=3)
    assert np.array_equal(train("rf", X, y, s, 3).policy_full(X),
                          train("rf", X, y, s, 3).policy_full(X))


def test_구조적_분할이_무작위보다_어렵다(데이터):
    # 왜: 이게 구조적 분할을 도입한 이유 자체다. 실측(seed 0, 비율 0.6)에서
    #     mlp가 0.877 -> 0.762로 떨어졌다. 두 값이 같아지면 구조적 분할이
    #     제 역할을 못 하고 있다는 신호다.
    X, y, keys = 데이터
    무작위 = train("mlp", X, y, make_split("random", keys, 0.6, seed=0), seed=0)
    구조적 = train("mlp", X, y, make_split("structural", keys, 0.6, seed=0), seed=0)
    assert 구조적.test_acc < 무작위.test_acc


def test_훈련_레이블_수가_분할과_맞는다(데이터):
    X, y, keys = 데이터
    s = make_split("structural", keys, 0.6, seed=0)
    assert train("rf", X, y, s, 0).n_train_labels == len(s.train_idx) == 6 * 61


def test_학습_시간이_기록되고_10초를_넘지_않는다(데이터):
    # 왜 이 상한인가: 실측 0.1~0.9초다. 10초를 넘으면 설정이 잘못된 것이다.
    X, y, keys = 데이터
    m = train("rf", X, y, make_split("random", keys, 0.6, seed=0), seed=0)
    assert 0.0 < m.fit_seconds < 10.0


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_파라미터_수를_셀_수_있다(데이터, kind):
    X, y, keys = 데이터
    assert train(kind, X, y, make_split("random", keys, 1.0, seed=0), 0).n_params() > 0


def test_모르는_종류는_예외():
    with pytest.raises(ValueError):
        make_estimator("xgboost", seed=0)
