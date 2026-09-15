"""이 파일은 최대화 편향 프로브와 다섯 알고리즘의 실제 학습 결과를 검증한다
입력: RULES_V1, 세션 DP 해, TabularAgent
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.agents.loop import make_env, run_episodes
from blackjack_rl.agents.tabular import TabularAgent
from blackjack_rl.dp.exact import evaluate_policy
from blackjack_rl.eval.bias import PROBE_KEYS, maxq_minus_vstar, probe_vstar_mean
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_V1
from blackjack_rl.starts import NaturalDeal
from blackjack_rl.state import LEGAL, REACHABLE_KEYS, Q_SHAPE

# 이 저장소의 DP가 직접 푼 값들. 학습 정책이 DP 최적보다 좋으면 버그다.
EV_INITIAL = -0.005108
EV_DEALER_MIMIC = -0.056746      # 딜러 모방(17 미만 히트) 정책의 DP 정확 EV


# ── 프로브 집합 자체 ──

def test_프로브는_50칸이고_전부_도달_가능하다():
    assert len(PROBE_KEYS) == 50
    도달 = set(REACHABLE_KEYS)
    for key in PROBE_KEYS:
        assert key in 도달
        assert key.is_soft == 0
        assert key.can_double == 0 and key.can_split == 0
        assert 12 <= key.total <= 16
    assert len(set(PROBE_KEYS)) == 50


def test_프로브의_최적가치_평균은_DP가_준_값이다(dp):
    # 왜 이 값인가: 실제로 solve_optimal(RULES_V1)을 돌려 뽑은 -0.3079509031이다.
    #     이 숫자가 흔들리면 프로브 집합 정의가 바뀐 것이므로 즉시 알아야 한다.
    assert probe_vstar_mean(dp.V) == pytest.approx(-0.3079509031, abs=1e-9)


def test_프로브_칸의_V는_전부_유한하다(dp):
    for key in PROBE_KEYS:
        assert np.isfinite(dp.V[key])


# ── 프로브 함수의 산수 ──

def test_Q가_V와_같으면_편향은_0이다(dp):
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float64)
    for key in PROBE_KEYS:
        q[key] = np.where(LEGAL[key], float(dp.V[key]), -np.inf)
    assert maxq_minus_vstar(q, dp.V) == pytest.approx(0.0, abs=1e-12)


def test_Q를_일정하게_부풀리면_그만큼_편향이_커진다(dp):
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float64)
    for key in PROBE_KEYS:
        q[key] = np.where(LEGAL[key], float(dp.V[key]) + 0.07, -np.inf)
    assert maxq_minus_vstar(q, dp.V) == pytest.approx(0.07, abs=1e-12)


def test_불법_행동의_마이너스무한대는_무시된다(dp):
    # 왜: Q 테이블의 불법 칸은 -inf다. max에 그대로 섞이면 결과가 -inf가 된다.
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float64)
    for key in PROBE_KEYS:
        q[key] = np.where(LEGAL[key], float(dp.V[key]), -np.inf)
    assert np.isfinite(maxq_minus_vstar(q, dp.V))


def test_키를_직접_넘길_수_있다(dp):
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float64)
    키 = PROBE_KEYS[:1]
    q[키[0]] = np.where(LEGAL[키[0]], float(dp.V[키[0]]) + 0.25, -np.inf)
    assert maxq_minus_vstar(q, dp.V, 키) == pytest.approx(0.25, abs=1e-12)


# ── 실제로 돌려 보는 통합 검증 ──

def _학습(algo, n, seed, *, step="sample_average", alpha=0.02, eps=None):
    """자연 딜로 n판 학습한 에이전트. eps를 주면 ε를 그 값으로 고정한다."""
    streams = make_streams(seed)
    kw = dict(algo=algo, step=step, alpha=alpha)
    if eps is None:
        kw.update(eps0=0.25, eps_final=0.02, eps_decay_at=int(n * 0.8))
    else:
        kw.update(eps0=eps, eps_final=eps, eps_decay_at=1)
    agent = TabularAgent(RULES_V1, streams, **kw)
    env = make_env(RULES_V1, streams)
    run_episodes(env, agent, NaturalDeal(), streams.explore, n)
    return agent


@pytest.mark.slow
def test_다섯_알고리즘이_10만판에_실제로_EV를_올린다(dp):
    """수동 실행 전용(약 35초). pytest -m slow 로만 돈다."""
    표 = {}
    for algo in ("mc", "q", "doubleq", "sarsa", "esarsa"):
        ev들, 편향들 = [], []
        for seed in (0, 1, 2):
            agent = _학습(algo, 100_000, seed)
            ev들.append(evaluate_policy(agent.greedy_full(), RULES_V1))
            편향들.append(maxq_minus_vstar(agent.q_eval_table(), dp.V))
        표[algo] = (float(np.mean(ev들)), float(np.mean(편향들)))
        print(f"{algo:8s} EV={표[algo][0]:+.5f} maxq-V*={표[algo][1]:+.4f}")

    for algo, (ev, 편향) in 표.items():
        # 왜 위 경계가 EV_INITIAL인가: DP 최적 EV보다 좋은 값은 물리적으로 불가능하다.
        #     여기서 걸리면 정산·보상 부호 버그다(설계서 §2.16 sanity_alarm과 같은 논리).
        assert ev < EV_INITIAL, f"{algo}가 DP 최적을 이겼다 — 버그다"
        # 왜 아래 경계가 -0.055인가: 10만판 × 시드 5개 실측에서 개별 시드 최악이
        #     mc의 -0.0479였고 TD 4종 최악은 doubleq의 -0.0358이었다. -0.055는
        #     딜러 모방(-0.056746)보다 확실히 낫다는 것을 보장하면서 여유를 둔 값이다.
        assert ev > -0.055, f"{algo}가 10만판에도 딜러 모방 수준을 못 넘었다"
        assert ev > EV_DEALER_MIMIC
        # 왜 0.06인가: 프로브 편향은 10만판 3시드 평균이 q -0.0169 / doubleq -0.0178 /
        #     sarsa -0.0264 / esarsa -0.0219 였고 개별 시드 최악이 -0.0466이었다.
        assert abs(편향) < 0.06, f"{algo}의 프로브 편향이 비정상이다: {편향}"

    # 왜 이 비교인가: 같은 판수·같은 시드·같은 ε 스케줄에서 TD 부트스트랩이
    #     몬테카를로보다 빨리 배운다는 것이 W5의 핵심 주장이다.
    #     10만판 실측에서 q가 mc를 5/5 시드로 이겼고 최소 마진이 0.008이었다.
    assert 표["q"][0] > 표["mc"][0] + 0.004


@pytest.mark.slow
def test_두_추정기_모두_V별에서_크게_벗어나지_않는다(dp):
    """블랙잭 규모에서 최대화 편향이 얼마나 되는지 재고 기록한다. 수동 실행(약 3~4분).

    **이 테스트는 q > doubleq 를 단언하지 않는다. 그 효과가 이 규모에서
    측정되지 않기 때문이다.** 100만판 · α 고정 0.02 · ε 고정 0.10 · 10시드로
    직접 재 본 결과(2026-09-16):

        q        평균 -0.00649  SD 0.01468  SE 0.00464
        doubleq  평균 -0.00868  SD 0.00984  SE 0.00311
        차이     +0.00218 ± 0.00377  ->  0.58 sigma
        시드별 q > doubleq : 5/10  (동전 던지기)

    방향 자체는 교과서와 같지만 0.58σ라 증거가 될 수 없다. 원인은 분명하다 —
    최대화 편향은 '행동 가치들이 비등하고 추정 잡음이 클 때' 생기는데,
    블랙잭은 대부분의 칸에서 Q가 충분히 벌어져 있고(하드 16 vs 10 같은 몇 칸이 예외)
    100만판이면 칸당 방문이 많아 잡음이 작다.

    현상 자체와 구현의 정확성은 `tests/test_tabular_target.py` 의 합성 실험이
    증명한다(Q러닝 +1.0248 vs 이론값 E[max of 4 N(0,1)]=1.0294, Double Q -0.0028).
    발표에서 '최대화 편향' 그림을 쓰려면 블랙잭 학습 곡선이 아니라 그 합성 실험을
    써야 한다. 이 사실을 여기 적어 두는 것이 이 테스트의 목적이다.
    """
    편향 = {}
    시드별 = {}
    for algo in ("q", "doubleq"):
        값들 = [maxq_minus_vstar(
            _학습(algo, 1_000_000, s, step="constant", alpha=0.02,
                 eps=0.10).q_eval_table(), dp.V) for s in (0, 1, 2, 3, 4)]
        시드별[algo] = 값들
        편향[algo] = float(np.mean(값들))
        print(f"{algo:8s} maxq-V* = {편향[algo]:+.4f}  시드별 {np.round(값들, 4)}")

    # 실제로 측정 가능한 것만 단언한다: 두 추정기 모두 V*에서 크게 벗어나지 않는다.
    # 폭주하는 편향(예: 부트스트랩이 발산해 +0.3씩 뜨는 경우)은 이 검사에 걸린다.
    for algo in ("q", "doubleq"):
        assert abs(편향[algo]) < 0.05, f"{algo}의 편향 {편향[algo]:+.4f}가 너무 크다"
        for v in 시드별[algo]:
            assert abs(v) < 0.10, f"{algo}의 한 시드가 {v:+.4f}로 튀었다"
