"""이 파일은 MC 에이전트가 30만 판으로 정말 학습하는지 끝에서 끝까지 검사한다.
입력: 고정 시드 20260916, 탐험적 시작, 표본평균 스텝사이즈.
출력: pytest 통과/실패."""

import numpy as np
import pytest

from blackjack_rl.agents.loop import make_env, run_episodes
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.dp.exact import evaluate_policy
from blackjack_rl.rng import make_streams
from blackjack_rl.starts import ExploringStarts, NaturalDeal
from blackjack_rl.state import (DOUBLE, HIT, KEY_INDEX, SPLIT, STAND,
                                StateKey, legal_actions)

N_EPISODES = 300_000
SEED = 20260916

# 실측: 무작위(합법 4개 균등) 정확 EV = -0.460235, DP 최적 = -0.005108.
# 같은 조건을 시드 5개(0,1,2,7,20260916)로 돌리면 -0.015421 ~ -0.010585 였다
# (seed 20260916은 -0.012074). 아래 구간은 그 폭의 3배 이상을 잡아 둔 것이다.
EV_최소 = -0.05
EV_최대 = -0.0052   # DP 최적보다 좋게 나오면 그건 학습이 아니라 버그다

# 이 칸들은 DP 최선과 차선의 차이가 커서 시드가 바뀌어도 뒤집히지 않는다(5시드 확인).
# 쓰지 않은 칸과 그 이유:
#   하드16 vs 10 — DP가 스탠드 -0.540430 / 히트 -0.539826 으로 갭 0.000604, 사실상 동점
#   하드13 vs 2, 8,8 vs 10 — 300k 학습에서 시드마다 뒤집힌다(실측)
#   동점 칸으로 학습 성공을 판정하면 안 된다.
확실한_칸 = (
    (StateKey(20, 0, 6, 0, 0, 0), STAND, "하드 20 vs 6은 스탠드"),
    (StateKey(19, 0, 6, 0, 0, 0), STAND, "하드 19 vs 6은 스탠드"),
    (StateKey(21, 0, 10, 0, 0, 0), STAND, "하드 21 vs 10은 스탠드"),
    (StateKey(5, 0, 10, 1, 0, 0), HIT, "하드 5 vs 10은 히트"),
    (StateKey(11, 0, 6, 1, 0, 0), DOUBLE, "하드 11 vs 6은 더블"),
    (StateKey(12, 1, 6, 1, 1, 0), SPLIT, "A,A vs 6은 스플릿"),
    (StateKey(20, 0, 6, 1, 1, 0), STAND, "10,10 vs 6은 스플릿하지 않는다"),
)


@pytest.fixture(scope="module")
def 학습결과(rules):
    """30만 판 학습을 모듈당 한 번만 돌린다(실측 5.0초, 63,118 eps/s)."""
    streams = make_streams(SEED)
    env = make_env(rules, streams)
    agent = TabularAgent(rules, streams, algo="mc", step="sample_average",
                         eps0=0.25, eps_final=0.02, eps_decay_at=N_EPISODES)
    sampler = ExploringStarts(rules)
    n_transitions = run_episodes(env, agent, sampler, streams.explore, N_EPISODES)
    return agent, sampler, n_transitions


def test_에피소드와_전이_개수가_말이_된다(학습결과):
    agent, _sampler, n_transitions = 학습결과
    assert agent.t == N_EPISODES
    assert agent.eps == pytest.approx(0.02)
    # 실측: 432,750개 = 에피소드당 1.4425개. 구간을 넉넉히 잡는다.
    assert N_EPISODES * 1.2 < n_transitions < N_EPISODES * 2.0


def test_EV가_무작위에서_확실히_올라온다(학습결과, rules):
    agent, _sampler, _n = 학습결과
    ev = evaluate_policy(agent.greedy_full(), rules)
    assert EV_최소 < ev < EV_최대


def test_학습한_정책은_DP_최적보다_좋을_수_없다(학습결과, rules, dp):
    """블랙잭 RL의 버그는 대부분 '결과가 좋아지는' 방향으로 나타난다.
    DP 최적을 넘으면 정산이나 크레딧 할당이 틀린 것이다."""
    agent, _sampler, _n = 학습결과
    ev = evaluate_policy(agent.greedy_full(), rules)
    assert ev < dp.ev_initial


@pytest.mark.parametrize("key,기대행동,설명", 확실한_칸)
def test_명백한_칸을_맞힌다(학습결과, key, 기대행동, 설명):
    agent, _sampler, _n = 학습결과
    고른것 = int(agent.greedy_full()[KEY_INDEX[key]])
    assert 고른것 == 기대행동, 설명


def test_탐험적_시작이_커버한_칸은_하나도_빠짐없이_방문된다(학습결과):
    agent, sampler, _n = 학습결과
    최소방문 = 10 ** 9
    for key in sampler.covered_keys:
        mask = legal_actions(key)
        for action in range(4):
            if mask[action]:
                방문 = int(agent.N[tuple(key) + (action,)])
                최소방문 = min(최소방문, 방문)
    # 실측: 1,590개 쌍 중 방문 0회는 0개, 최소 방문 106회, 평균 271.97회
    #   (전이 총합 432,750 / 1,590). 하한 50은 그 절반이다.
    assert 최소방문 >= 50


def test_커버하지_못한_칸은_사실상_학습되지_않는다(학습결과):
    """Task 7의 일치율 계산이 이 칸들을 '미결정'으로 빼야 하는 근거다.
    하드 20 can_double=1 은 10을 세 번 연속 스플릿해야만 도달하므로
    30만 판에서 실측 0회였다."""
    agent, _sampler, _n = 학습결과
    assert int(agent.N[StateKey(20, 0, 6, 1, 0, 0)].sum()) < 100
    assert int(agent.N[StateKey(4, 0, 6, 1, 0, 0)].sum()) < 100


def test_같은_시드는_같은_Q를_만든다(rules):
    """재현성. 짧게(2만 판) 두 번 돌려 바이트 단위로 같은지 본다."""
    def 한번(seed):
        streams = make_streams(seed)
        env = make_env(rules, streams)
        agent = TabularAgent(rules, streams, algo="mc", eps_decay_at=20_000)
        run_episodes(env, agent, ExploringStarts(rules), streams.explore, 20_000)
        return agent

    a = 한번(5)
    b = 한번(5)
    assert np.array_equal(a.Q, b.Q)
    assert np.array_equal(a.N, b.N)


def test_두_스텝사이즈가_서로_다른_표를_만든다(rules):
    """2x2 실험이 성립하려면 step 인자가 실제로 결과를 바꿔야 한다."""
    def 한번(step):
        streams = make_streams(5)
        env = make_env(rules, streams)
        agent = TabularAgent(rules, streams, algo="mc", step=step,
                             alpha=0.02, eps_decay_at=20_000)
        run_episodes(env, agent, ExploringStarts(rules), streams.explore, 20_000)
        return agent

    표본평균 = 한번("sample_average")
    고정알파 = 한번("constant")
    assert not np.array_equal(표본평균.Q, 고정알파.Q)
    # 왜 방문 횟수도 달라지는가: act()가 self.Q의 argmax를 보므로 갱신식이 다르면
    #   첫 그리디 선택부터 갈라지고 카드 소비도 갈라진다. 20,000 에피소드 실측에서
    #   N이 1,417칸 다르고 합계가 28,666 vs 28,656 이었다. '같아야 한다'고 쓰면 실패한다.
    assert not np.array_equal(표본평균.N, 고정알파.N)
    # 다만 총 결정 수는 거의 같다(같은 게임을 같은 판수만큼 한다).
    assert abs(int(표본평균.N.sum()) - int(고정알파.N.sum())) / 표본평균.N.sum() < 0.01


def test_자연딜_샘플러로도_같은_루프가_돈다(rules):
    """NaturalDeal().sample()은 None을 돌려주고 환경이 알아서 딜한다.
    루프에 분기를 하나도 두지 않아도 두 시작 분포가 모두 돈다."""
    streams = make_streams(11)
    env = make_env(rules, streams)
    agent = TabularAgent(rules, streams, algo="mc", eps_decay_at=5_000)
    n_transitions = run_episodes(env, agent, NaturalDeal(), streams.explore, 5_000)
    assert agent.t == 5_000
    assert n_transitions > 0


def test_진행_콜백이_정해진_횟수만큼_불린다(rules):
    streams = make_streams(1)
    env = make_env(rules, streams)
    agent = TabularAgent(rules, streams, algo="mc", eps_decay_at=1_000)
    본것 = []
    run_episodes(env, agent, ExploringStarts(rules), streams.explore, 1_000,
                 progress=본것.append, progress_every=250)
    assert 본것 == [250, 500, 750, 1000]


def test_스냅샷_지점마다_콜백이_에피소드_번호와_함께_불린다(rules):
    """Task 10의 runner가 프레임을 찍는 방법이다. runner가 루프를 다시 쓰지 않아도
    되게 하는 것이 이 인자의 존재 이유다."""
    streams = make_streams(1)
    env = make_env(rules, streams)
    agent = TabularAgent(rules, streams, algo="mc", eps_decay_at=1_000)
    찍힌것 = []
    run_episodes(env, agent, ExploringStarts(rules), streams.explore, 1_000,
                 snapshot_at={1, 7, 1_000},
                 on_snapshot=lambda ep: 찍힌것.append((ep, agent.t)))
    # 왜 (ep, agent.t)가 같아야 하는가: 스냅샷은 그 에피소드의 observe가 끝난 뒤에
    #   찍혀야 한다. 먼저 찍으면 프레임 하나만큼 뒤처진 Q를 저장하게 된다.
    assert 찍힌것 == [(1, 1), (7, 7), (1_000, 1_000)]
