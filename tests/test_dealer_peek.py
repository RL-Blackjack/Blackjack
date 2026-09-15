"""이 파일은 peeked=True 분포가 '딜러 BJ 아님' 조건부 분포와 같은지 검증한다.
입력: 업카드 A(1)와 10, RULES_V1, 업카드마다 100만 판(딜러 BJ 판은 기각) 시뮬레이션.
출력: pytest 통과/실패 (재정규화 항등식 + 3시그마 교차검증을 모두 통과해야 한다).
"""
import numpy as np
import pytest

from blackjack_rl.cards import make_shoe
from blackjack_rl.dealer import dealer_has_blackjack, play_dealer
from blackjack_rl.dp.dealer_dp import BUST_INDEX, CARD_PROB, OUTCOME_LABELS, dealer_outcome_dist
from blackjack_rl.hand import Hand
from blackjack_rl.rules import RULES_V1

N_SIMS = 1_000_000


def simulate_dealer_dist(up, rules, n, seed, *, peeked):
    """play_dealer만 써서 딜러 최종 결과의 빈도를 센다.

    peeked=True면 딜러가 블랙잭인 판을 버린다(기각 샘플링) = '딜러 BJ 아님' 조건부.
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
    """이항분포 표준오차의 3배. p는 해석해 확률(참값), n은 시뮬 판수."""
    se = np.sqrt(p * (1.0 - p) / n)
    return 3.0 * se


def blackjack_hole_prob(up):
    """업카드가 up일 때 홀카드가 블랙잭을 완성할 확률."""
    if up == 1:
        return CARD_PROB[10]   # A 위에 10이 오면 BJ
    if up == 10:
        return CARD_PROB[1]    # 10 위에 A가 오면 BJ
    return 0.0


@pytest.mark.parametrize("up", [1, 10])
def test_peek_renormalization_identity(up):
    """재정규화가 빠지면 즉시 걸리는 대수 항등식.

    (BJ 아님 조건부) x (BJ 아닐 확률) + (BJ) x (21 확정) = (피크 안 한 분포)
    """
    unpeeked = np.asarray(dealer_outcome_dist(up, RULES_V1, peeked=False))
    peeked = np.asarray(dealer_outcome_dist(up, RULES_V1, peeked=True))
    p_bj = blackjack_hole_prob(up)

    just_21 = np.zeros(6, dtype=np.float64)
    just_21[4] = 1.0   # 인덱스 4 = '21'

    rebuilt = p_bj * just_21 + (1.0 - p_bj) * peeked
    assert np.allclose(rebuilt, unpeeked, atol=1e-12), (
        f"up={up}\n피크X={unpeeked}\n피크O={peeked}\n재조립={rebuilt}"
    )
    # 재정규화를 빠뜨리면 합이 1보다 작아진다(up=A면 9/13, up=10이면 12/13)
    assert abs(float(peeked.sum()) - 1.0) < 1e-12
    # BJ를 빼냈으므로 21 확률은 반드시 줄어든다
    assert peeked[4] < unpeeked[4]


@pytest.mark.parametrize("up", [2, 3, 4, 5, 6, 7, 8, 9])
def test_peek_changes_nothing_for_other_upcards(up):
    """A와 10이 아닌 업카드는 피크해도 분포가 그대로다."""
    unpeeked = np.asarray(dealer_outcome_dist(up, RULES_V1, peeked=False))
    peeked = np.asarray(dealer_outcome_dist(up, RULES_V1, peeked=True))
    assert np.allclose(unpeeked, peeked, atol=1e-15)


@pytest.mark.parametrize("up", [1, 10])
def test_dealer_peek(up):
    """peeked=True 해석해 vs '딜러 BJ 판을 버린' 시뮬 100만 판, 6칸 전부 3시그마 이내."""
    exact = np.asarray(dealer_outcome_dist(up, RULES_V1, peeked=True))
    sim = simulate_dealer_dist(up, RULES_V1, N_SIMS, seed=2000 + up, peeked=True)
    tol = three_sigma(exact, N_SIMS)
    diff = np.abs(sim - exact)
    detail = "\n".join(
        f"  {OUTCOME_LABELS[i]:>4}: 해석해={exact[i]:.6f} 시뮬={sim[i]:.6f} "
        f"차이={diff[i]:.6f} 3시그마={tol[i]:.6f}"
        for i in range(6)
    )
    assert np.all(diff <= tol), f"피크 조건화 업카드 {up}에서 3시그마를 벗어났다:\n{detail}"
