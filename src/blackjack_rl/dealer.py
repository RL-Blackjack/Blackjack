"""이 파일은 딜러가 규칙대로 카드를 뽑는 과정(S17/H17, 피크, 미추첨)을 시뮬레이션한다.
입력: draw()를 가진 슈, 딜러 업카드와 홀카드, RuleSet, 살아있는 플레이어 손 유무.
출력: 딜러가 최종적으로 들고 있는 카드 리스트([업카드, 홀카드, 추가 카드...]).
"""
from __future__ import annotations

from blackjack_rl.hand import Hand
from blackjack_rl.rules import RuleSet


def dealer_has_blackjack(up: int, hole: int) -> bool:
    """딜러의 첫 두 장이 블랙잭인가."""
    return Hand(cards=[up, hole]).is_blackjack


def _dealer_hits(total: int, soft: bool, rules: RuleSet) -> bool:
    """딜러가 한 장 더 뽑아야 하는가."""
    if total < 17:
        return True
    if total == 17 and soft and rules.dealer_hits_soft_17:
        # 왜: H17 규칙에서만 소프트 17을 한 번 더 친다. S17이면 여기서 선다.
        return True
    # 왜: 22 이상(버스트)도 이 줄로 내려와 False가 되므로 별도 버스트 검사가 필요 없다.
    return False


def play_dealer(shoe, up: int, hole: int, rules: RuleSet, any_live_hand: bool) -> list[int]:
    """딜러의 손을 끝까지 진행하고 최종 카드 리스트를 돌려준다."""
    cards = [up, hole]

    if dealer_has_blackjack(up, hole):
        # 왜: 딜러 블랙잭이면 라운드가 즉시 끝나므로 카드를 더 뽑지 않는다.
        return cards

    if not any_live_hand:
        # 왜: 살아있는 플레이어 손이 하나도 없으면 딜러는 홀카드만 까고 끝낸다.
        #     여기서 카드를 뽑아 버리면 슈 소모량이 달라져 카운팅 통계가 통째로 틀어진다.
        return cards

    while True:
        hand = Hand(cards=cards)
        if not _dealer_hits(hand.total, hand.is_soft, rules):
            break
        cards.append(shoe.draw())

    return cards
