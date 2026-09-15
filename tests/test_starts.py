import numpy as np
import pytest

from blackjack_rl.starts import NaturalDeal, StartState


def test_자연딜_샘플러는_항상_None을_반환한다():
    rng = np.random.default_rng(0)
    sampler = NaturalDeal()
    assert all(sampler.sample(rng) is None for _ in range(100))


def test_시작상태는_플레이어카드와_딜러업카드를_담는다():
    s = StartState(player_cards=[10, 6], dealer_up=10)
    assert s.player_cards == [10, 6]
    assert s.dealer_up == 10
    assert s.forced_first_action is None


def test_첫_행동을_강제로_지정할_수_있다():
    s = StartState(player_cards=[1, 7], dealer_up=9, forced_first_action=1)
    assert s.forced_first_action == 1
