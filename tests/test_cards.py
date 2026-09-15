"""이 파일은 카드 값 계산과 슈 3종이 규칙대로 동작하는지 확인한다.
입력: blackjack_rl.cards의 함수·클래스와 blackjack_rl.rng의 난수 스트림.
출력: 10의 비율, 슈 소진, 컷카드, Hi-Lo 카운트, 재생 순서에 대한 pytest 결과.
"""

import numpy as np
import pytest

from blackjack_rl.cards import (
    CARD_VALUES,
    CARDS_PER_DECK,
    FiniteShoe,
    InfiniteShoe,
    ReplayShoe,
    Shoe,
    card_value,
    hilo_tag,
    make_shoe,
)
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_SHOE, RULES_V1, RuleSet


def test_카드값_테이블이_한_덱을_나타낸다():
    assert len(CARD_VALUES) == 13
    assert CARD_VALUES.count(10) == 4      # 10, J, Q, K
    assert CARD_VALUES.count(1) == 1       # A
    assert sum(CARD_VALUES) == 85
    assert CARDS_PER_DECK == 52


def test_card_value는_에이스_표기를_1로_되돌린다():
    assert card_value(11) == 1   # dealer_up 표기의 A
    assert card_value(1) == 1
    assert card_value(10) == 10
    assert card_value(7) == 7
    with pytest.raises(ValueError):
        card_value(0)
    with pytest.raises(ValueError):
        card_value(12)
    with pytest.raises(ValueError):
        card_value(-3)


def test_hilo_태그가_교과서대로다():
    기대값 = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1, 11: -1}
    for 카드, 태그 in 기대값.items():
        assert hilo_tag(카드) == 태그


def test_한_덱_전체의_hilo_합은_0이다():
    # 왜: Hi-Lo는 균형 카운트다. 한 덱을 다 세면 반드시 0이어야 한다.
    한덱 = [값 for 값 in CARD_VALUES for _ in range(4)]
    assert sum(hilo_tag(카드) for 카드 in 한덱) == 0


def test_무한슈에서_10이_나올_확률은_4_13이다():
    슈 = InfiniteShoe(make_streams(0).deal)
    n = 1_000_000
    십_횟수 = 0
    for _ in range(n):
        if 슈.draw() == 10:
            십_횟수 += 1
    비율 = 십_횟수 / n
    assert abs(비율 - 4 / 13) < 0.003


def test_무한슈가_뽑는_값은_항상_1에서_10이다():
    슈 = InfiniteShoe(make_streams(1).deal)
    본값 = set()
    for _ in range(10_000):
        본값.add(슈.draw())
    assert 본값 == set(range(1, 11))


def test_무한슈는_카운팅이_무의미하다():
    슈 = InfiniteShoe(make_streams(2).deal)
    for _ in range(100):
        슈.draw()
    assert 슈.running_count == 0
    assert 슈.true_count == 0.0
    assert 슈.decks_remaining == float("inf")
    assert 슈.maybe_shuffle() is False


def test_유한슈는_정확히_52곱하기덱수만큼_나오고_소진된다():
    슈 = FiniteShoe(decks=2, penetration=0.75, rng=make_streams(3).deal)
    뽑은카드 = [슈.draw() for _ in range(104)]
    assert len(뽑은카드) == 104
    assert 슈.cards_remaining == 0
    with pytest.raises(RuntimeError):
        슈.draw()


def test_유한슈의_구성은_실제_덱과_같다():
    슈 = FiniteShoe(decks=1, penetration=0.75, rng=make_streams(4).deal)
    뽑은카드 = [슈.draw() for _ in range(52)]
    assert 뽑은카드.count(10) == 16
    assert 뽑은카드.count(1) == 4
    for 값 in range(2, 10):
        assert 뽑은카드.count(값) == 4


def test_컷카드에_도달하면_maybe_shuffle이_참을_돌려준다():
    슈 = FiniteShoe(decks=1, penetration=0.75, rng=make_streams(5).deal)
    # 52 * 0.75 = 39장째가 컷카드다.
    for _ in range(38):
        슈.draw()
    assert 슈.maybe_shuffle() is False
    슈.draw()                       # 39장째
    assert 슈.maybe_shuffle() is True
    assert 슈.cards_remaining == 52  # 섞은 뒤에는 처음으로 돌아간다
    assert 슈.running_count == 0
    assert 슈.decks_remaining == 1.0


def test_러닝카운트가_뽑은_카드의_태그_합과_같다():
    슈 = FiniteShoe(decks=2, penetration=0.75, rng=make_streams(6).deal)
    합 = 0
    for _ in range(50):
        카드 = 슈.draw()
        합 += hilo_tag(카드)
        assert 슈.running_count == 합


def test_한_슈를_다_뽑으면_러닝카운트가_0으로_돌아온다():
    슈 = FiniteShoe(decks=1, penetration=1.0, rng=make_streams(7).deal)
    for _ in range(52):
        슈.draw()
    assert 슈.running_count == 0


def test_트루카운트는_남은_덱수로_나눈_값이다():
    슈 = FiniteShoe(decks=4, penetration=0.75, rng=make_streams(8).deal)
    for _ in range(104):
        슈.draw()
    assert 슈.decks_remaining == pytest.approx(2.0)
    assert 슈.true_count == pytest.approx(슈.running_count / 2.0)


def test_유한슈의_잘못된_설정은_예외():
    rng = make_streams(9).deal
    with pytest.raises(ValueError):
        FiniteShoe(decks=0, penetration=0.75, rng=rng)
    with pytest.raises(ValueError):
        FiniteShoe(decks=9, penetration=0.75, rng=rng)
    with pytest.raises(ValueError):
        FiniteShoe(decks=4, penetration=0.0, rng=rng)


def test_리플레이슈는_주어진_순서_그대로_돌려준다():
    카드들 = [10, 1, 5, 5, 10, 2]
    슈 = ReplayShoe(카드들)
    assert [슈.draw() for _ in range(6)] == 카드들
    with pytest.raises(RuntimeError):
        슈.draw()


def test_리플레이슈는_절대_섞지_않는다():
    슈 = ReplayShoe(np.array([2, 3, 4, 5], dtype=np.int8))
    assert 슈.maybe_shuffle() is False
    assert 슈.draw() == 2
    assert 슈.maybe_shuffle() is False
    assert 슈.draw() == 3


def test_리플레이슈도_hilo를_센다():
    슈 = ReplayShoe([5, 10, 7])
    슈.draw()                      # +1
    슈.draw()                      # -1
    슈.draw()                      # 0
    assert 슈.running_count == 0


def test_리플레이슈의_잘못된_카드는_예외():
    with pytest.raises(ValueError):
        ReplayShoe([])
    with pytest.raises(ValueError):
        ReplayShoe([1, 2, 99])


def test_make_shoe가_규칙에_맞는_슈를_고른다():
    rng = make_streams(10).deal
    assert isinstance(make_shoe(RULES_V1, rng), InfiniteShoe)
    슈 = make_shoe(RULES_SHOE, rng)
    assert isinstance(슈, FiniteShoe)
    assert 슈.cards_remaining == 52 * RULES_SHOE.decks


def test_세_슈가_모두_Shoe_프로토콜을_만족한다():
    rng = make_streams(11).deal
    for 슈 in [
        InfiniteShoe(rng),
        FiniteShoe(decks=4, penetration=0.75, rng=rng),
        ReplayShoe([2, 3, 4]),
    ]:
        assert isinstance(슈, Shoe)


def test_공개_API가_패키지_최상단에서_바로_보인다():
    import blackjack_rl

    assert blackjack_rl.RuleSet is RuleSet
    assert blackjack_rl.make_shoe is make_shoe
    assert blackjack_rl.card_value is card_value
    assert hasattr(blackjack_rl, "make_streams")
    assert hasattr(blackjack_rl, "RULES_V1")
