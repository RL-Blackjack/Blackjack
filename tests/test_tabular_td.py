"""이 파일은 TabularAgent가 다섯 알고리즘을 '같은 루프'로 학습하는지 검증한다
입력: RULES_V1, make_streams, BlackjackEnv
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.agents.tabular import TD_ALGOS, TabularAgent
from blackjack_rl.cards import InfiniteShoe
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.returns import Transition, compute_transitions
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import HIT, LEGAL, REACHABLE_KEYS, STAND, StateKey

HARD16 = StateKey(16, 0, 10, 0, 0, 0)
HARD13 = StateKey(13, 0, 6, 0, 0, 0)
CHILD1 = StateKey(18, 0, 6, 1, 0, 0)


def 에이전트(algo, seed=0, **kw):
    return TabularAgent(RULES_V1, make_streams(seed), algo=algo, **kw)


# ── 표 구성 ──

def test_doubleq만_두번째_표를_가진다():
    assert 에이전트("doubleq").QB is not None
    assert 에이전트("doubleq").NB is not None
    for algo in ("mc", "q", "sarsa", "esarsa"):
        assert 에이전트(algo).QB is None
        assert 에이전트(algo).NB is None


def test_q_eval과_visit_table은_doubleq에서_두_표를_합친다():
    ag = 에이전트("doubleq")
    ag.Q[HARD16] = [0.2, 0.6, -np.inf, -np.inf]
    ag.QB[HARD16] = [0.4, -0.2, -np.inf, -np.inf]
    assert ag.q_eval(HARD16)[STAND] == pytest.approx(0.3)
    assert ag.q_eval(HARD16)[HIT] == pytest.approx(0.2)
    assert ag.q_eval_table()[HARD16][STAND] == pytest.approx(0.3)
    # 왜 방문수는 평균이 아니라 합인가: 한 라운드의 갱신이 둘 중 한 표에만 들어가므로
    #   '이 칸을 몇 번 봤나'는 두 표의 합이다. 평균을 쓰면 미결정 칸 판정이 틀어진다.
    ag.N[HARD16 + (STAND,)] = 7
    ag.NB[HARD16 + (STAND,)] = 5
    assert int(ag.visit_table()[HARD16 + (STAND,)]) == 12


def test_불법_행동은_q_eval_table에서도_마이너스무한대다():
    # 왜: (-inf + -inf)/2 = -inf 여야 한다. nan이 되면 저장한 npz의 재현성 점검이 깨진다.
    ag = 에이전트("doubleq")
    t = ag.q_eval_table()
    assert bool(np.isneginf(t[~LEGAL]).all())
    assert not bool(np.isnan(t).any())


# ── 갱신이 올바른 칸에 들어가는가 ──

def test_observe는_방문한_칸만_갱신한다():
    ag = 에이전트("q")
    before = ag.Q.copy()
    tr = Transition(HARD16, HIT, -1.0, (), -1.0, True)
    ag.observe([tr])
    바뀐칸 = np.argwhere(ag.Q != before)
    assert len(바뀐칸) == 1
    assert tuple(바뀐칸[0]) == HARD16 + (HIT,)
    assert ag.N[HARD16 + (HIT,)] == 1


def test_sample_average는_첫_갱신에서_타깃값_그대로가_된다():
    # 왜: N=1이면 α=1/1=1이므로 Q ← 타깃. 스텝 크기 공식이 맞는지 한 줄로 드러난다.
    ag = 에이전트("q", step="sample_average")
    ag.observe([Transition(HARD16, STAND, -0.5, (), -0.5, True)])
    assert ag.Q[HARD16 + (STAND,)] == pytest.approx(-0.5)


def test_constant는_알파만큼만_움직인다():
    ag = 에이전트("q", step="constant", alpha=0.25)
    시작 = float(ag.Q[HARD16 + (STAND,)])
    ag.observe([Transition(HARD16, STAND, -1.0, (), -1.0, True)])
    기대 = 시작 + 0.25 * (-1.0 - 시작)
    assert ag.Q[HARD16 + (STAND,)] == pytest.approx(기대)


def test_mc는_mc_return을_쓰고_TD는_안_쓴다():
    # 부트스트랩할 후계자 값을 0.9로 박아 두고 mc_return은 -1.0으로 둔다.
    tr = Transition(HARD13, HIT, 0.0, (CHILD1,), -1.0, False)
    mc = 에이전트("mc")
    mc.observe([tr])
    assert mc.Q[HARD13 + (HIT,)] == pytest.approx(-1.0)

    td = 에이전트("q")
    td.Q[CHILD1] = [0.9, 0.1, -0.3, -np.inf]
    td.observe([tr])
    assert td.Q[HARD13 + (HIT,)] == pytest.approx(0.9)


def test_doubleq는_두_표를_대략_반반씩_갱신한다():
    ag = 에이전트("doubleq", seed=5)
    s = make_streams(5)
    env = BlackjackEnv(RULES_V1, InfiniteShoe(s.deal), s)
    for _ in range(3000):
        ag.observe(compute_transitions(env.play_round(ag.act)))
    a, b = int(ag.N.sum()), int(ag.NB.sum())
    # 왜 2500인가: 라운드당 결정 수는 약 1.28이라 3,000라운드면 갱신 약 3,832회다
    #   (실측 N=1,919 / NB=1,913). 딜러·플레이어 블랙잭으로 결정이 0회인 라운드가
    #   약 8.5% 있으므로 하한은 여유 있게 2,500으로 잡는다.
    assert a + b > 2500
    # 왜 이 구간인가: 동전 던지기 p=0.5, 표본 약 3,800회면 표준편차가 0.008이다.
    #     실측 0.5008. ±0.05는 6시그마 밖이라 시드가 바뀌어도 안 깨진다.
    assert 0.45 < a / (a + b) < 0.55


# ── "차이는 _target 하나뿐"의 증거 ──

def test_학습_전에는_세_알고리즘의_행동_수열이_완전히_같다():
    # 왜: Q 초기화(init 스트림)도 ε 스케줄도 탐험 난수(explore 스트림)도 공유하므로,
    #     observe를 한 번도 부르지 않으면 q/sarsa/esarsa는 문자 그대로 같은 행동을 낸다.
    #     이것이 "공정 비교를 어떻게 보장했나"의 실행 가능한 증거다.
    #     (doubleq는 표를 둘 만드느라 init 스트림을 두 번 쓰므로 여기서 제외한다.)
    수열 = {}
    for algo in ("q", "sarsa", "esarsa"):
        ag = 에이전트(algo, seed=99)
        수열[algo] = [ag.act(HARD16, LEGAL[HARD16], None) for _ in range(200)]
    assert 수열["q"] == 수열["sarsa"] == 수열["esarsa"]


def test_모르는_알고리즘은_observe에서_예외다():
    ag = 에이전트("q")
    ag.algo = "qlearning"
    with pytest.raises(ValueError):
        ag.observe([Transition(HARD13, HIT, 0.0, (CHILD1,), 0.0, False)])


# ── 정책 추출 ──

def test_greedy_full은_길이_610의_int8이고_불법을_고르지_않는다():
    for algo in TD_ALGOS:
        ag = 에이전트(algo)
        pol = ag.greedy_full()
        assert pol.shape == (len(REACHABLE_KEYS),)
        assert pol.dtype == np.int8
        assert set(np.unique(pol)).issubset({0, 1, 2, 3})
        for i, key in enumerate(REACHABLE_KEYS):
            assert LEGAL[key][pol[i]]


def test_max_q는_합법_행동_중_최대다():
    ag = 에이전트("q")
    ag.Q[HARD16] = [-0.9, -0.2, -np.inf, -np.inf]
    assert ag.max_q(HARD16) == pytest.approx(-0.2)
