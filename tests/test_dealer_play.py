"""이 파일은 play_dealer가 딜러 규칙(S17/H17, 하드16 힛, 버스트, 미추첨)을 지키는지 본다.
입력: 미리 정해 둔 카드를 순서대로 돌려주는 가짜 슈와 RULES_V1(S17)/RULES_H17(H17).
출력: pytest 통과/실패.
"""
import pytest

from blackjack_rl.dealer import dealer_has_blackjack, play_dealer
from blackjack_rl.hand import Hand
from blackjack_rl.rules import RULES_H17, RULES_V1


class FakeShoe:
    """미리 정해 둔 카드를 순서대로 돌려주는 시험용 슈. 몇 장 뽑혔는지도 센다."""

    def __init__(self, cards):
        self.cards = list(cards)
        self.drawn = 0

    def draw(self) -> int:
        card = self.cards[self.drawn]
        self.drawn += 1
        return card


def test_s17_stands_on_soft_17():
    shoe = FakeShoe([5, 5, 5])
    cards = play_dealer(shoe, up=1, hole=6, rules=RULES_V1, any_live_hand=True)
    assert cards == [1, 6]              # A,6 = 소프트 17 -> S17이면 그대로 선다
    assert shoe.drawn == 0


def test_s17_stands_on_three_card_soft_17():
    shoe = FakeShoe([5, 5, 5])
    cards = play_dealer(shoe, up=1, hole=2, rules=RULES_V1, any_live_hand=True)
    # A,2 = 소프트 13 -> 힛 -> 5를 받아 소프트 18이 되어 멈춘다
    assert cards == [1, 2, 5]
    assert Hand(cards=cards).total == 18
    assert shoe.drawn == 1


def test_h17_hits_soft_17():
    shoe = FakeShoe([5, 10])
    cards = play_dealer(shoe, up=1, hole=6, rules=RULES_H17, any_live_hand=True)
    # A,6 = 소프트 17 -> H17이면 힛 -> 5를 받아 하드 12 -> 또 힛 -> 10을 받아 22 버스트
    assert cards == [1, 6, 5, 10]
    assert Hand(cards=cards).total == 22
    assert shoe.drawn == 2


def test_hard_16_must_hit():
    shoe = FakeShoe([5])
    cards = play_dealer(shoe, up=10, hole=6, rules=RULES_V1, any_live_hand=True)
    assert cards == [10, 6, 5]
    assert Hand(cards=cards).total == 21
    assert shoe.drawn == 1


def test_bust_over_21():
    shoe = FakeShoe([10])
    cards = play_dealer(shoe, up=10, hole=6, rules=RULES_V1, any_live_hand=True)
    assert Hand(cards=cards).total == 26
    assert Hand(cards=cards).is_bust is True
    assert shoe.drawn == 1


def test_stands_at_18_hits_at_12():
    shoe = FakeShoe([6])
    assert play_dealer(shoe, up=10, hole=8, rules=RULES_V1, any_live_hand=True) == [10, 8]
    assert shoe.drawn == 0

    shoe2 = FakeShoe([6])
    assert play_dealer(shoe2, up=10, hole=2, rules=RULES_V1, any_live_hand=True) == [10, 2, 6]
    assert shoe2.drawn == 1


def test_no_draw_when_no_live_hand():
    """★ 플레이어가 전원 버스트면 딜러는 카드를 한 장도 뽑지 않는다."""
    shoe = FakeShoe([5, 5, 5])
    cards = play_dealer(shoe, up=10, hole=6, rules=RULES_V1, any_live_hand=False)
    assert cards == [10, 6]
    assert shoe.drawn == 0   # 여기가 틀리면 슈 소모량이 달라져 카운트 통계가 깨진다


def test_blackjack_stops_immediately():
    assert dealer_has_blackjack(1, 10) is True
    assert dealer_has_blackjack(10, 1) is True
    assert dealer_has_blackjack(10, 10) is False
    assert dealer_has_blackjack(1, 1) is False

    shoe = FakeShoe([5])
    cards = play_dealer(shoe, up=1, hole=10, rules=RULES_V1, any_live_hand=True)
    assert cards == [1, 10]
    assert shoe.drawn == 0


@pytest.mark.parametrize("rules", [RULES_V1, RULES_H17])
def test_never_stops_below_17(rules):
    """어떤 규칙에서도 최종 합은 17 이상이거나 버스트다."""
    for hole in range(1, 11):
        shoe = FakeShoe([10, 10, 10, 10, 10, 10])
        cards = play_dealer(shoe, up=2, hole=hole, rules=rules, any_live_hand=True)
        total = Hand(cards=cards).total
        assert total >= 17, f"hole={hole}, cards={cards}, total={total}"
