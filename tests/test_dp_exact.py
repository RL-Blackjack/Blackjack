"""이 파일은 무한덱 정확 DP의 결과가 이론값·상식과 맞는지 검증한다
입력: RULES_V1 규칙과 blackjack_rl.dp.exact의 함수들
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.dp.exact import (
    action_values,
    add_card,
    evaluate_policy,
    evaluate_stochastic,
    greedy_policy_full,
    solve_optimal,
)
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import (
    DOUBLE,
    HIT,
    Q_SHAPE,
    REACHABLE_KEYS,
    SPLIT,
    STAND,
    StateKey,
)


@pytest.fixture(scope="module")
def dp():
    return solve_optimal(RULES_V1)


# ── 카드 더하기 기본 동작 ──

def test_add_card_에이스_처리():
    assert add_card(11, 1, 1) == (12, 1)      # A + A = 소프트 12
    assert add_card(13, 1, 9) == (12, 0)      # A,2 + 9 = 하드 12
    assert add_card(20, 0, 1) == (21, 0)      # 20 + A = 하드 21
    assert add_card(11, 1, 10) == (21, 1)     # A + 10 = 소프트 21
    assert add_card(16, 0, 10) == (26, 0)     # 버스트는 그대로 26으로 돌려준다


# ── DPResult 모양 ──

def test_결과_배열의_모양과_지문(dp):
    assert dp.Q.shape == Q_SHAPE
    assert dp.V.shape == Q_SHAPE[:-1]
    assert dp.gap.shape == Q_SHAPE[:-1]
    assert dp.rules_fp == RULES_V1.fingerprint()


def test_불법_행동은_nan이다(dp):
    # can_double=0, can_split=0, split_depth=0 인 하드 16 vs 10
    q = dp.Q[16, 0, 10, 0, 0, 0]
    assert not np.isnan(q[STAND])
    assert not np.isnan(q[HIT])
    assert np.isnan(q[DOUBLE])
    assert np.isnan(q[SPLIT])


def test_gap은_음수가_아니다(dp):
    finite = dp.gap[np.isfinite(dp.gap)]
    assert finite.size > 0
    assert float(finite.min()) >= 0.0


# ── 핵심 숫자: 하우스엣지 ──

def test_하우스엣지가_알려진_구간에_들어온다(dp):
    # S17 / DAS / BJ 3:2 기본전략의 알려진 하우스엣지 ≈ -0.5%
    assert -0.006 < dp.ev_initial < -0.004


# ── 상식과 맞는가 ──

def test_하드20_대_6은_스탠드가_최선이다():
    q = action_values(StateKey(20, 0, 6, 1, 0), RULES_V1)
    assert int(np.nanargmax(q)) == STAND
    assert np.isnan(q[SPLIT])


def test_하드5_대_10은_히트가_최선이다():
    q = action_values(StateKey(5, 0, 10, 1, 0), RULES_V1)
    assert int(np.nanargmax(q)) == HIT


def test_AA_대_6은_스플릿이_최선이다():
    q = action_values(StateKey(12, 1, 6, 1, 1), RULES_V1)
    assert int(np.nanargmax(q)) == SPLIT


def test_action_values는_페어가_아닌_합계에_can_split을_허용하지_않는다():
    with pytest.raises(ValueError):
        action_values(StateKey(15, 0, 6, 1, 1), RULES_V1)


# ── 정책 평가 ──

def test_최적정책_평가가_DP_최적값과_일치한다(dp):
    policy = greedy_policy_full(dp)
    ev = evaluate_policy(policy, RULES_V1)
    # 표(StateKey) 단위 정책은 깊이별 최적 정책의 부분집합이므로 절대 넘을 수 없다
    assert ev <= dp.ev_initial + 1e-12
    # split_depth 별칭으로 생기는 오차는 0.01%p 미만이어야 한다
    assert abs(ev - dp.ev_initial) < 1e-4


def test_항상_스탠드_정책은_훨씬_나쁘다():
    policy = np.full(len(REACHABLE_KEYS), STAND, dtype=np.int8)
    ev = evaluate_policy(policy, RULES_V1)
    assert ev < -0.10


def test_확률적_평가는_결정적_평가의_일반화다():
    pi = np.zeros((len(REACHABLE_KEYS), 4), dtype=np.float64)
    pi[:, STAND] = 1.0
    ev_sto = evaluate_stochastic(pi, RULES_V1)
    ev_det = evaluate_policy(np.full(len(REACHABLE_KEYS), STAND, dtype=np.int8), RULES_V1)
    assert abs(ev_sto - ev_det) < 1e-12


def test_탐험이_섞이면_EV가_나빠진다(dp):
    policy = greedy_policy_full(dp)
    eps = 0.10
    pi = np.zeros((len(REACHABLE_KEYS), 4), dtype=np.float64)
    for i, key in enumerate(REACHABLE_KEYS):
        legal = [STAND, HIT]
        if key.can_double:
            legal.append(DOUBLE)
        if key.can_split:
            legal.append(SPLIT)
        for a in legal:
            pi[i, a] += eps / len(legal)
        pi[i, int(policy[i])] += 1.0 - eps
    ev_behavior = evaluate_stochastic(pi, RULES_V1)
    assert ev_behavior < evaluate_policy(policy, RULES_V1)
