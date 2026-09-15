"""이 파일은 TabularAgent의 부품 하나하나(ε 스케줄, 행동 선택, 갱신식, 사영)를 검사한다.
입력: 손으로 만든 Transition과 고정 시드.
출력: pytest 통과/실패."""

import numpy as np
import pytest

from blackjack_rl.agents.tabular import (LEGAL_ROWS, TabularAgent,
                                         eps_greedy_probs)
from blackjack_rl.env import Ctx
from blackjack_rl.returns import Transition
from blackjack_rl.rng import make_streams
from blackjack_rl.state import (DOUBLE, HIT, KEY_INDEX, REACHABLE_KEYS, SPLIT,
                                STAND, StateKey, legal_actions)

CTX = Ctx(true_count=0.0, decks_left=float("inf"), hand_index=0, n_hands=1, bet=1.0)
K_HIT만 = StateKey(16, 0, 10, 0, 0, 0)      # 합법 = 스탠드, 히트
K_더블가능 = StateKey(11, 0, 6, 1, 0, 0)     # 합법 = 스탠드, 히트, 더블
K_페어 = StateKey(16, 0, 10, 1, 1, 0)        # 합법 = 네 가지 전부


def 새_에이전트(rules, seed=0, **kw):
    return TabularAgent(rules, make_streams(seed), algo="mc", **kw)


def test_처음_Q는_불법에_음의_무한대_합법에_작은_노이즈다(rules):
    agent = 새_에이전트(rules)
    assert agent.Q[K_HIT만][DOUBLE] == -np.inf
    assert agent.Q[K_HIT만][SPLIT] == -np.inf
    assert abs(agent.Q[K_HIT만][STAND]) <= 1e-6
    assert agent.N.sum() == 0
    assert agent.QB is None
    assert agent.NB is None
    assert agent.t == 0


def test_엡실론은_선형으로_줄고_바닥에서_멈춘다(rules):
    agent = 새_에이전트(rules, eps0=0.25, eps_final=0.02, eps_decay_at=1000)
    assert agent.current_eps() == pytest.approx(0.25)
    agent.t = 500
    assert agent.current_eps() == pytest.approx(0.135)
    agent.t = 1000
    assert agent.current_eps() == pytest.approx(0.02)
    agent.t = 999_999
    assert agent.current_eps() == pytest.approx(0.02)


def test_엡실론이_0이면_항상_Q가_가장_큰_행동을_고른다(rules):
    agent = 새_에이전트(rules, eps0=0.0, eps_final=0.0, eps_decay_at=1)
    agent.Q[K_더블가능] = np.array([0.1, 0.9, 0.5, -np.inf])
    mask = legal_actions(K_더블가능)
    for _ in range(50):
        assert agent.act(K_더블가능, mask, CTX) == HIT


def test_엡실론이_1이면_합법_행동_네_개가_모두_나온다(rules):
    agent = 새_에이전트(rules, eps0=1.0, eps_final=1.0, eps_decay_at=1)
    mask = legal_actions(K_페어)
    본것 = set()
    for _ in range(2000):
        본것.add(agent.act(K_페어, mask, CTX))
    assert 본것 == {STAND, HIT, DOUBLE, SPLIT}


def test_불법_행동은_절대_고르지_않는다(rules):
    agent = 새_에이전트(rules, eps0=0.5, eps_final=0.5, eps_decay_at=1)
    mask = legal_actions(K_HIT만)
    for _ in range(3000):
        assert agent.act(K_HIT만, mask, CTX) in (STAND, HIT)


def test_표본평균은_지금까지_받은_리턴의_진짜_평균이다(rules):
    """1/N 스텝사이즈의 정의 그 자체. 초기 노이즈가 첫 갱신에서 완전히 지워진다."""
    agent = 새_에이전트(rules, step="sample_average")
    for 리턴 in (1.0, -1.0, 0.5):
        tr = Transition(K_HIT만, STAND, 리턴, (), 리턴, True)
        agent.observe([tr])
    assert agent.N[K_HIT만][STAND] == 3
    assert agent.Q[K_HIT만][STAND] == pytest.approx(0.5 / 3.0)
    assert agent.t == 3


def test_고정_알파는_한_번에_알파만큼만_움직인다(rules):
    agent = 새_에이전트(rules, step="constant", alpha=0.1)
    이전 = float(agent.Q[K_HIT만][STAND])
    tr = Transition(K_HIT만, STAND, 1.0, (), 1.0, True)
    agent.observe([tr])
    assert agent.Q[K_HIT만][STAND] == pytest.approx(이전 + 0.1 * (1.0 - 이전))
    assert agent.Q[K_HIT만][STAND] == pytest.approx(0.1, abs=1e-6)


def test_MC는_보상이_아니라_mc_return을_쓴다(rules):
    """스플릿 결정의 reward는 0이고 진짜 크레딧은 mc_return에 들어 있다.
    reward를 쓰면 스플릿이 영원히 0점으로 보인다."""
    agent = 새_에이전트(rules, step="sample_average")
    자식키 = StateKey(13, 0, 10, 1, 0, 0)
    tr = Transition(K_페어, SPLIT, 0.0, (자식키, 자식키), 2.0, False)
    agent.observe([tr])
    assert agent.Q[K_페어][SPLIT] == pytest.approx(2.0)


def test_에피소드_수는_전이_개수가_아니라_라운드_수로_센다(rules):
    agent = 새_에이전트(rules)
    tr = Transition(K_HIT만, STAND, 1.0, (), 1.0, True)
    agent.observe([tr, tr, tr])
    assert agent.t == 1


def test_전이가_하나도_없는_라운드도_에피소드로_센다(rules):
    """딜러 블랙잭이면 결정이 한 번도 없다. 그래도 ε 스케줄은 진행해야 한다."""
    agent = 새_에이전트(rules)
    agent.observe([])
    assert agent.t == 1


def test_greedy_full은_길이_610의_int8이고_전부_합법이다(rules):
    agent = 새_에이전트(rules)
    policy = agent.greedy_full()
    assert policy.shape == (len(REACHABLE_KEYS),)
    assert policy.shape == (610,)
    assert policy.dtype == np.int8
    for i, key in enumerate(REACHABLE_KEYS):
        assert legal_actions(key)[policy[i]]


def test_greedy_full은_학습한_칸을_그대로_반영한다(rules):
    agent = 새_에이전트(rules)
    agent.Q[K_더블가능] = np.array([0.1, 0.2, 0.9, -np.inf])
    assert int(agent.greedy_full()[KEY_INDEX[K_더블가능]]) == DOUBLE


def test_엡실론_그리디_분포는_행마다_합이_1이고_불법은_0이다(rules):
    agent = 새_에이전트(rules)
    pi = agent.eps_greedy_dist(0.2)
    assert pi.shape == (610, 4)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert float(pi[~LEGAL_ROWS].max()) == 0.0
    assert float(pi[~LEGAL_ROWS].min()) == 0.0


def test_엡실론_그리디_분포의_칸_값이_공식과_같다(rules):
    agent = 새_에이전트(rules)
    agent.Q[K_HIT만] = np.array([0.9, 0.1, -np.inf, -np.inf])
    pi = agent.eps_greedy_dist(0.2)
    행 = pi[KEY_INDEX[K_HIT만]]
    # 합법 2개, eps=0.2 -> 각 0.1, 그리디에 0.8 추가
    assert 행[STAND] == pytest.approx(0.9)
    assert 행[HIT] == pytest.approx(0.1)


def test_두_엡실론그리디_구현이_같은_벡터를_준다(rules):
    """eps_greedy_dist(610x4 팬시 인덱싱)와 eps_greedy_probs(한 상태)가 갈라지면
    Expected SARSA의 타깃과 ev_behavior 곡선이 서로 다른 정책을 말하게 된다."""
    agent = 새_에이전트(rules)
    agent.Q[K_페어] = np.array([0.3, -0.1, 0.05, 0.7])
    한줄 = eps_greedy_probs(agent.Q[K_페어], legal_actions(K_페어), 0.2)
    전체 = agent.eps_greedy_dist(0.2)[KEY_INDEX[K_페어]]
    assert np.allclose(한줄, 전체)


def test_max_q는_합법_행동_중_최댓값이다(rules):
    agent = 새_에이전트(rules)
    agent.Q[K_더블가능] = np.array([0.1, 0.7, 0.3, -np.inf])
    assert agent.max_q(K_더블가능) == pytest.approx(0.7)


def test_없는_알고리즘_스텝사이즈_인코더는_거부한다(rules):
    streams = make_streams(0)
    with pytest.raises(ValueError):
        TabularAgent(rules, streams, algo="dqn")
    with pytest.raises(ValueError):
        TabularAgent(rules, streams, algo="mc", step="cosine")
    with pytest.raises(ValueError):
        # 왜: s2(트루카운트 포함)는 계획서 ④ W9의 몫이다. 지금 받아 두면
        #     설정에만 존재하고 동작이 없는 플래그가 된다.
        TabularAgent(rules, streams, algo="mc", encoder="s2")


def test_같은_시드면_탐험_난수까지_똑같다(rules):
    a = 새_에이전트(rules, seed=7, eps0=0.5, eps_final=0.5, eps_decay_at=1)
    b = 새_에이전트(rules, seed=7, eps0=0.5, eps_final=0.5, eps_decay_at=1)
    mask = legal_actions(K_페어)
    선택_a = [a.act(K_페어, mask, CTX) for _ in range(200)]
    선택_b = [b.act(K_페어, mask, CTX) for _ in range(200)]
    assert 선택_a == 선택_b
