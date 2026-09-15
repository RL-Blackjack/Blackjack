"""이 파일은 Hand의 합계·소프트 판정·페어·블랙잭·상태키를 전수 검증한다.
입력: 카드 1~10으로 만들 수 있는 6장 이하의 모든 조합(8,007가지)과 RULES_V1.
출력: pytest 통과/실패 (독립적으로 다시 계산한 값과 Hand 속성이 전부 같아야 통과).
"""
from itertools import combinations_with_replacement

import pytest

from blackjack_rl.hand import Hand, dealer_up_index
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import StateKey


def brute_total(cards):
    """Hand 코드를 전혀 쓰지 않고 처음부터 다시 계산한다.

    에이스를 0장, 1장, ... n장 11로 써 보고 21 이하인 것 중 가장 큰 값을 고른다.
    21 이하가 하나도 없으면 전부 1로 센 값(= 버스트 합)을 쓴다.
    """
    n_ace = cards.count(1)
    low = sum(cards)
    candidates = [low + 10 * k for k in range(n_ace + 1)]
    ok = [t for t in candidates if t <= 21]
    if ok:
        total = max(ok)
    else:
        total = low
    return total, total > low


# 6장 이하 모든 조합. 순서는 합계에 영향이 없으므로 중복조합으로 충분하다.
ALL_HANDS = [
    list(combo)
    for size in range(1, 7)
    for combo in combinations_with_replacement(range(1, 11), size)
]


def test_all_hands_total_and_soft():
    assert len(ALL_HANDS) == 8007
    for cards in ALL_HANDS:
        want_total, want_soft = brute_total(cards)
        hand = Hand(cards=list(cards))
        assert hand.total == want_total, f"total 불일치: {cards}"
        assert hand.is_soft == want_soft, f"is_soft 불일치: {cards}"
        assert hand.is_bust == (want_total > 21), f"is_bust 불일치: {cards}"


@pytest.mark.parametrize(
    "cards, total, soft",
    [
        ([1, 1], 12, True),          # 함정: 에이스 두 장은 소프트 12
        ([1, 1, 1], 13, True),       # 함정: 에이스 세 장은 소프트 13
        ([1, 9, 1], 21, True),       # 함정: 소프트 21 (에이스 하나만 11)
        ([1, 10, 10], 21, False),    # 함정: 하드 21 (에이스를 1로 써야 함)
        ([1, 10], 21, True),         # 블랙잭은 소프트 21
        ([1, 5, 5, 10], 21, False),  # 에이스가 도중에 1로 내려앉는 경우
        ([1, 6], 17, True),          # 소프트 17
        ([1, 6, 10], 17, False),     # 소프트 17이 하드 17로 바뀐다
        ([10, 6], 16, False),
        ([10, 10, 10], 30, False),   # 버스트는 하드
    ],
)
def test_trap_cases(cards, total, soft):
    hand = Hand(cards=list(cards))
    assert hand.total == total
    assert hand.is_soft is soft


def test_is_pair_treats_face_cards_as_ten():
    assert Hand(cards=[10, 10]).is_pair is True   # 10/J/Q/K는 전부 10으로 들어온다
    assert Hand(cards=[8, 8]).is_pair is True
    assert Hand(cards=[1, 1]).is_pair is True
    assert Hand(cards=[7, 8]).is_pair is False
    assert Hand(cards=[10, 10, 10]).is_pair is False   # 3장은 페어가 아니다


def test_blackjack_requires_two_cards_and_no_split():
    assert Hand(cards=[1, 10]).is_blackjack is True
    assert Hand(cards=[10, 1]).is_blackjack is True
    assert Hand(cards=[1, 10], from_split=True).is_blackjack is False
    assert Hand(cards=[1, 5, 5]).is_blackjack is False
    assert Hand(cards=[10, 10]).is_blackjack is False


def test_dealer_up_index():
    assert dealer_up_index(1) == 11
    assert dealer_up_index(11) == 11
    assert dealer_up_index(2) == 2
    assert dealer_up_index(10) == 10
    with pytest.raises(ValueError):
        dealer_up_index(0)
    with pytest.raises(ValueError):
        dealer_up_index(12)


def test_state_key_fields():
    hand = Hand(cards=[8, 8])
    key = hand.state_key(dealer_up=1, rules=RULES_V1, hands_in_round=1)
    assert key == StateKey(
        total=16, is_soft=0, dealer_up=11, can_double=1, can_split=1, split_depth=0
    )


def test_state_key_soft_hand_and_no_double_after_hit():
    hand = Hand(cards=[1, 6, 3])   # 소프트 20, 3장이므로 더블 불가
    key = hand.state_key(dealer_up=10, rules=RULES_V1, hands_in_round=1)
    assert key == StateKey(
        total=20, is_soft=1, dealer_up=10, can_double=0, can_split=0, split_depth=0
    )


def test_state_key_rejects_bust_hand():
    hand = Hand(cards=[10, 10, 10])
    with pytest.raises(ValueError):
        hand.state_key(dealer_up=5, rules=RULES_V1, hands_in_round=1)


def test_state_key_hides_split_depth_by_default():
    hand = Hand(cards=[5, 4], split_depth=2, from_split=True)
    key = hand.state_key(dealer_up=6, rules=RULES_V1, hands_in_round=3)
    assert key.split_depth == 0   # include_split_depth_in_state=False가 기본


def test_can_split_limits():
    deep = Hand(cards=[8, 8], split_depth=3, from_split=True)
    assert deep.can_split(RULES_V1, hands_in_round=1) is False   # 깊이 상한 3

    crowded = Hand(cards=[8, 8])
    assert crowded.can_split(RULES_V1, hands_in_round=4) is False  # 라운드 4손 상한

    split_ace = Hand(cards=[1, 1], from_split_ace=True)
    assert split_ace.can_split(RULES_V1, hands_in_round=1) is False  # resplit_aces=False

    fresh_ace = Hand(cards=[1, 1])
    assert fresh_ace.can_split(RULES_V1, hands_in_round=1) is True


def test_can_double_rules():
    assert Hand(cards=[5, 6]).can_double(RULES_V1) is True
    assert Hand(cards=[5, 6, 2]).can_double(RULES_V1) is False        # 3장이면 불가
    assert Hand(cards=[5, 6], doubled=True).can_double(RULES_V1) is False
    assert Hand(cards=[5, 6], from_split=True).can_double(RULES_V1) is True   # DAS=True
    split_ace = Hand(cards=[1, 7], from_split=True, from_split_ace=True)
    assert split_ace.can_double(RULES_V1) is False   # 에이스 스플릿은 1장만 받고 끝
