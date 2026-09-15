"""이 파일은 타깃 함수 _target()이 알고리즘 다섯 종을 정확히 구현했는지 검증한다
입력: 손으로 만든 Q/QB 테이블과 Transition
출력: pytest 통과/실패
"""

import collections

import numpy as np
import pytest

# 왜: _target은 밑줄로 시작하지만 설계서 §2.12가 이름을 그렇게 못박았다.
#     같은 패키지 안의 테스트이므로 그대로 가져다 쓴다.
from blackjack_rl.agents.tabular import (ALGOS, TD_ALGOS, _target,
                                         eps_greedy_probs, successor_value)
from blackjack_rl.returns import Transition
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import HIT, SPLIT, STAND, StateKey, new_q_table

# 8,8 vs 딜러 6 (설계서 §3.4의 손계산 예제와 같은 상황)
PAIR = StateKey(16, 0, 6, 1, 1, 0)        # 합법 4개
CHILD1 = StateKey(18, 0, 6, 1, 0, 0)      # 합법 3개 (S/H/D)
CHILD2 = StateKey(11, 0, 6, 1, 0, 0)      # 합법 3개 (S/H/D)
ACE_PAIR = StateKey(12, 1, 6, 1, 1, 0)    # A,A vs 6
HARD13 = StateKey(13, 0, 6, 0, 0, 0)      # 합법 2개 (S/H)
FOUR_ACTIONS = StateKey(16, 0, 10, 1, 1, 0)


@pytest.fixture
def tables():
    """손으로 값을 박아 넣은 Q/QB 한 쌍."""
    rng = np.random.default_rng(0)
    Q = new_q_table(RULES_V1, rng)
    QB = new_q_table(RULES_V1, rng)
    Q[CHILD1] = [0.4, 0.1, -0.3, -np.inf]     # 최선은 STAND=0.4
    Q[CHILD2] = [-0.2, -0.5, -0.9, -np.inf]   # 최선은 STAND=-0.2
    QB[CHILD1] = [-0.7, 0.9, 0.2, -np.inf]
    QB[CHILD2] = [0.6, -0.1, 0.3, -np.inf]
    return Q, QB


def test_알고리즘_목록은_다섯개고_TD는_넷이다():
    assert TD_ALGOS == ("q", "doubleq", "sarsa", "esarsa")
    assert ALGOS == ("mc", "q", "doubleq", "sarsa", "esarsa")


# ── MC 분기가 _target 맨 앞에 있는가 ──

def test_mc는_next_keys가_있어도_mc_return만_쓴다(tables):
    """_target이 두 번 정의되면 가장 먼저 깨지는 자리다. mc 분기는 맨 앞에 있어야 한다."""
    Q, QB = tables
    tr = Transition(PAIR, SPLIT, 0.0, (CHILD1, CHILD2), -1.75, False)
    assert _target("mc", tr, Q, QB, 0.1, np.random.default_rng(0)) == pytest.approx(-1.75)


# ── 종단 처리 ──

def test_종단_전이는_보상을_그대로_돌려준다(tables):
    Q, QB = tables
    tr = Transition(CHILD1, STAND, 1.0, (), 1.0, True)
    for algo in TD_ALGOS:
        assert _target(algo, tr, Q, QB, 0.1, np.random.default_rng(0)) == 1.0


def test_에이스_스플릿은_terminal이_False여도_보상만_쓴다(tables):
    # 왜: returns.py는 A,A 스플릿에서 자식이 결정 없이 끝나므로
    #     terminal=False인데 next_keys=() 인 전이를 만든다(실측 전체의 0.43%).
    #     terminal 플래그로 분기하면 여기서 반드시 틀린다.
    Q, QB = tables
    tr = Transition(ACE_PAIR, SPLIT, -1.5, (), -1.5, False)
    assert tr.terminal is False
    for algo in TD_ALGOS:
        assert _target(algo, tr, Q, QB, 0.1, np.random.default_rng(0)) == -1.5


# ── 스플릿 백업: 후계자 두 개를 '더한다' (설계서 §3.4) ──

def test_Q러닝_스플릿은_두_후계자의_max를_더한다(tables):
    Q, QB = tables
    tr = Transition(PAIR, SPLIT, 0.0, (CHILD1, CHILD2), -1.0, False)
    # 0.0 + max(Q[CHILD1]) + max(Q[CHILD2]) = 0.0 + 0.4 + (-0.2) = 0.2
    assert _target("q", tr, Q, QB, 0.1) == pytest.approx(0.2)


def test_Q러닝_히트는_후계자_하나의_max다(tables):
    Q, QB = tables
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)
    assert _target("q", tr, Q, QB, 0.1) == pytest.approx(0.4)


def test_스플릿은_평균이_아니라_합이다(tables):
    # 왜: 평균으로 짜면 베팅 2단위짜리 스플릿의 가치가 절반이 되어
    #     "쪼개면 손해"라는 틀린 표를 만든다. 합과 평균을 명시적으로 갈라 둔다.
    Q, QB = tables
    tr = Transition(PAIR, SPLIT, 0.0, (CHILD1, CHILD2), -1.0, False)
    합 = _target("q", tr, Q, QB, 0.1)
    평균 = (0.4 + (-0.2)) / 2.0
    assert 합 != pytest.approx(평균)


# ── Double Q: argmax는 Q에서, 값은 QB에서 ──

def test_doubleq는_argmax는_Q_값은_QB에서_읽는다(tables):
    Q, QB = tables
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)
    # Q[CHILD1]의 argmax는 STAND(0) → QB[CHILD1][0] = -0.7
    assert _target("doubleq", tr, Q, QB, 0.1) == pytest.approx(-0.7)


def test_doubleq_스플릿도_두_후계자를_더한다(tables):
    Q, QB = tables
    tr = Transition(PAIR, SPLIT, 0.0, (CHILD1, CHILD2), -1.0, False)
    # QB[CHILD1][argmax Q[CHILD1]=0] + QB[CHILD2][argmax Q[CHILD2]=0]
    assert _target("doubleq", tr, Q, QB, 0.1) == pytest.approx(-0.7 + 0.6)


# ── Expected SARSA: ε-greedy 분포의 기댓값 ──

def test_엡실론그리디_분포는_불법행동에_0을_준다(tables):
    Q, _ = tables
    mask = np.array([True, True, True, False])
    probs = eps_greedy_probs(Q[CHILD1], mask, 0.3)
    # 합법 3개에 0.3/3=0.1씩, 최선(STAND)에 +0.7
    assert probs == pytest.approx([0.8, 0.1, 0.1, 0.0])
    assert probs.sum() == pytest.approx(1.0)


def test_esarsa는_손계산과_같다(tables):
    Q, QB = tables
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)
    # 0.8*0.4 + 0.1*0.1 + 0.1*(-0.3) = 0.32 + 0.01 - 0.03 = 0.30
    assert _target("esarsa", tr, Q, QB, 0.3) == pytest.approx(0.30)


def test_esarsa는_엡실론0이면_Q러닝과_같다(tables):
    Q, QB = tables
    tr = Transition(PAIR, SPLIT, 0.0, (CHILD1, CHILD2), -1.0, False)
    assert _target("esarsa", tr, Q, QB, 0.0) == pytest.approx(
        _target("q", tr, Q, QB, 0.0)
    )


# ── SARSA: 같은 ε-greedy 분포에서 '한 개를 뽑는다' ──

def test_sarsa는_엡실론그리디_분포에서_뽑는다(tables):
    Q, QB = tables
    rng = np.random.default_rng(1)
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)
    값들 = [_target("sarsa", tr, Q, QB, 0.3, rng) for _ in range(20000)]
    셈 = collections.Counter(np.round(값들, 4))
    # 왜 이 구간인가: 이론 빈도는 0.8/0.1/0.1이고 20,000표본의 표준오차는
    #     각각 0.0028/0.0021이다. ±0.02는 7시그마 밖이라 안전하다.
    #     실측(seed 1): 0.4가 15,997회(0.7999), 0.1이 1,966회, -0.3이 2,037회.
    assert 셈[0.4] / 20000 == pytest.approx(0.8, abs=0.02)
    assert 셈[0.1] / 20000 == pytest.approx(0.1, abs=0.02)
    assert 셈[-0.3] / 20000 == pytest.approx(0.1, abs=0.02)


def test_sarsa는_rng가_없으면_예외다(tables):
    Q, QB = tables
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)
    with pytest.raises(ValueError):
        _target("sarsa", tr, Q, QB, 0.3, None)


# ── 최대화 편향의 기계적 근거 ──

def test_최대화_편향은_Q러닝에만_생긴다():
    """참값이 전부 0인데 추정치에 잡음만 있을 때 무엇이 일어나는가."""
    g = np.random.default_rng(20260916)
    Q = new_q_table(RULES_V1, g)
    QB = new_q_table(RULES_V1, g)
    tr = Transition(StateKey(13, 0, 10, 0, 0, 0), HIT, 0.0,
                    (FOUR_ACTIONS,), 0.0, False)
    q_타깃, d_타깃 = [], []
    for _ in range(2000):
        Q[FOUR_ACTIONS] = g.normal(0.0, 1.0, 4)
        QB[FOUR_ACTIONS] = g.normal(0.0, 1.0, 4)
        q_타깃.append(_target("q", tr, Q, QB, 0.1))
        d_타깃.append(_target("doubleq", tr, Q, QB, 0.1))
    # 왜 이 구간인가: 표준정규 4개의 최댓값 기댓값은 이론값 1.0294이고
    #     2,000표본 표준오차가 0.016이다. 실측 1.0248. (0.90, 1.15)는 5시그마 밖.
    assert 0.90 < float(np.mean(q_타깃)) < 1.15
    # 왜 이 구간인가: Double Q의 기댓값은 이론적으로 정확히 0이고
    #     2,000표본 표준오차가 0.023이다. 실측 -0.0028. 0.10은 4시그마 밖.
    assert abs(float(np.mean(d_타깃))) < 0.10


def test_모르는_알고리즘은_예외다(tables):
    Q, QB = tables
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)
    with pytest.raises(ValueError):
        _target("qlearning", tr, Q, QB, 0.1)
    with pytest.raises(ValueError):
        successor_value("mc", CHILD1, Q, QB, 0.1, None)
