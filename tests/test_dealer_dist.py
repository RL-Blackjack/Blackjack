"""이 파일은 딜러 결과 확률의 해석해와 play_dealer 시뮬레이션을 3시그마로 교차검증한다.
입력: 업카드 1(A)~10, RULES_V1, 업카드마다 100만 판의 시뮬레이션.
출력: pytest 통과/실패 (업카드 10종 × 결과 6칸이 전부 3시그마 이내여야 통과).
"""
import numpy as np
import pytest

from blackjack_rl.cards import make_shoe
from blackjack_rl.dealer import dealer_has_blackjack, play_dealer
from blackjack_rl.dp.dealer_dp import (
    BUST_INDEX,
    OUTCOME_LABELS,
    dealer_bust_prob,
    dealer_outcome_dist,
)
from blackjack_rl.hand import Hand
from blackjack_rl.rules import RULES_V1

N_SIMS = 1_000_000


def simulate_dealer_dist(up, rules, n, seed, *, peeked):
    """play_dealer만 써서 딜러 최종 결과의 빈도를 센다.

    peeked=True면 딜러가 블랙잭인 판을 버린다(기각 샘플링) = '딜러 BJ 아님' 조건부.
    반환값은 길이 6 배열 [17,18,19,20,21,bust]의 상대빈도.
    """
    rng = np.random.default_rng(seed)
    shoe = make_shoe(rules, rng)
    counts = np.zeros(6, dtype=np.int64)
    done = 0
    while done < n:
        hole = shoe.draw()
        if peeked and dealer_has_blackjack(up, hole):
            continue
        cards = play_dealer(shoe, up, hole, rules, any_live_hand=True)
        total = Hand(cards=cards).total
        if total > 21:
            counts[BUST_INDEX] += 1
        else:
            assert total >= 17, f"딜러가 17 미만에서 멈췄다: {cards}"
            counts[total - 17] += 1
        done += 1
    return counts / n


def three_sigma(p, n):
    """이항분포 표준오차의 3배. p는 해석해 확률(참값), n은 시뮬 판수.

    시뮬 상대빈도는 Binomial(n, p)/n 이므로 표준오차는 sqrt(p*(1-p)/n) 이다.
    """
    se = np.sqrt(p * (1.0 - p) / n)
    return 3.0 * se


def test_dist_shape_and_sum():
    for up in range(1, 11):
        for peeked in (False, True):
            dist = dealer_outcome_dist(up, RULES_V1, peeked=peeked)
            assert dist.shape == (6,)
            assert dist.dtype == np.float64
            # 모든 결과가 양의 확률이어야 3시그마 허용오차가 0이 되지 않는다
            assert np.all(dist > 0.0), f"up={up}, peeked={peeked}, dist={dist}"
            assert abs(float(dist.sum()) - 1.0) < 1e-12, f"합이 1이 아니다: {dist.sum()}"


def test_dist_is_read_only():
    dist = dealer_outcome_dist(5, RULES_V1, peeked=False)
    with pytest.raises(ValueError):
        dist[0] = 0.5


def test_upcard_out_of_range_raises():
    with pytest.raises(ValueError):
        dealer_outcome_dist(0, RULES_V1, peeked=False)
    with pytest.raises(ValueError):
        dealer_outcome_dist(11, RULES_V1, peeked=False)


def test_bust_prob_matches_dist():
    for up in range(1, 11):
        dist = dealer_outcome_dist(up, RULES_V1, peeked=False)
        assert dealer_bust_prob(up, RULES_V1) == pytest.approx(float(dist[BUST_INDEX]))


@pytest.mark.parametrize("up", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_dealer_dist(up):
    """해석해 vs play_dealer 시뮬 100만 판, 6칸 전부 3시그마 이내."""
    exact = np.asarray(dealer_outcome_dist(up, RULES_V1, peeked=False))
    sim = simulate_dealer_dist(up, RULES_V1, N_SIMS, seed=1000 + up, peeked=False)
    tol = three_sigma(exact, N_SIMS)
    diff = np.abs(sim - exact)
    detail = "\n".join(
        f"  {OUTCOME_LABELS[i]:>4}: 해석해={exact[i]:.6f} 시뮬={sim[i]:.6f} "
        f"차이={diff[i]:.6f} 3시그마={tol[i]:.6f}"
        for i in range(6)
    )
    assert np.all(diff <= tol), f"업카드 {up}에서 3시그마를 벗어났다:\n{detail}"
