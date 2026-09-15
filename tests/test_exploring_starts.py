"""이 파일은 ExploringStarts가 도달 가능한 (상태, 행동)을 빠짐없이 균등하게 뽑는지 검사한다.
입력: 없음(REACHABLE_KEYS와 정본 규칙만 쓴다).
출력: pytest 통과/실패."""

import numpy as np

from blackjack_rl.hand import Hand
from blackjack_rl.starts import ExploringStarts, NaturalDeal, StartState, cards_for_key
from blackjack_rl.state import REACHABLE_KEYS, StateKey, legal_actions


def test_합법_쌍을_1590개_만든다(rules):
    sampler = ExploringStarts(rules)
    assert sampler.n_pairs == 1590
    assert len(sampler.pairs) == 1590


def test_커버한_키와_못한_키를_합치면_610개다(rules):
    sampler = ExploringStarts(rules)
    assert len(sampler.covered_keys) == 580
    assert len(sampler.uncovered_keys) == 30
    전체 = set(sampler.covered_keys) | set(sampler.uncovered_keys)
    assert 전체 == set(REACHABLE_KEYS)


def test_못_만드는_손은_정확히_세_모양뿐이다(rules):
    """하드4/하드20은 두 장이면 반드시 페어라 can_split=0을 만들 수 없고,
    소프트21은 두 장이면 [1,10]=블랙잭이라 결정 자체가 없다."""
    sampler = ExploringStarts(rules)
    모양들 = set()
    for key in sampler.uncovered_keys:
        모양들.add((key.total, key.is_soft, key.can_double, key.can_split))
    assert 모양들 == {(4, 0, 1, 0), (20, 0, 1, 0), (21, 1, 1, 0)}


def test_모든_쌍의_카드가_정확히_그_키를_만든다(rules):
    """카드 복원이 한 칸이라도 틀리면 학습이 엉뚱한 칸을 갱신한다. 1590개를 전부 본다."""
    sampler = ExploringStarts(rules)
    for pair in sampler.pairs:
        hand = Hand(cards=list(pair.cards))
        assert hand.is_blackjack is False
        assert hand.state_key(pair.key.dealer_up, rules, 1) == pair.key


def test_모든_쌍의_행동이_합법이다(rules):
    sampler = ExploringStarts(rules)
    for pair in sampler.pairs:
        assert legal_actions(pair.key)[pair.action]


def test_커버한_키의_합법_행동을_하나도_빠뜨리지_않는다(rules):
    sampler = ExploringStarts(rules)
    있는_쌍 = {(pair.key, pair.action) for pair in sampler.pairs}
    for key in sampler.covered_keys:
        mask = legal_actions(key)
        for action in range(4):
            if mask[action]:
                assert (key, action) in 있는_쌍


def test_에이스_업카드는_카드값_1로_넘긴다(rules):
    """StateKey는 에이스를 11로 쓰지만 env는 카드값(1)을 받는다.
    11을 그대로 넘기면 딜러 손 합계가 11로 계산돼 딜러 블랙잭 판정이 통째로 틀어진다."""
    sampler = ExploringStarts(rules)
    for pair in sampler.pairs:
        assert 1 <= pair.dealer_up_card <= 10
        if pair.key.dealer_up == 11:
            assert pair.dealer_up_card == 1
        else:
            assert pair.dealer_up_card == pair.key.dealer_up


def test_sample은_StartState를_돌려주고_같은_시드면_같다(rules):
    sampler = ExploringStarts(rules)
    s1 = sampler.sample(np.random.default_rng(3))
    s2 = sampler.sample(np.random.default_rng(3))
    assert isinstance(s1, StartState)
    assert s1.player_cards == s2.player_cards
    assert s1.dealer_up == s2.dealer_up
    assert s1.forced_first_action == s2.forced_first_action
    assert s1.forced_first_action is not None


def test_많이_뽑으면_모든_쌍이_고르게_나온다(rules):
    """균등 분포가 깨지면 어떤 칸은 영원히 학습되지 않는다."""
    sampler = ExploringStarts(rules)
    rng = np.random.default_rng(0)
    센다 = {}
    for _ in range(159_000):   # 쌍당 평균 100회
        s = sampler.sample(rng)
        열쇠 = (tuple(s.player_cards), s.dealer_up, s.forced_first_action)
        센다[열쇠] = 센다.get(열쇠, 0) + 1
    assert len(센다) == sampler.n_pairs
    # 왜 이 구간인가: 평균 100회, 표준편차 약 10회다. seed 0의 실측은 최소 70 / 최대 138로
    #   이론 분포와 맞는다. 50~160은 그보다 두 배 넓어, sample()이 rng를 조금 다르게
    #   소비하도록 바뀌어도 깨지지 않는다.
    assert min(센다.values()) >= 50
    assert max(센다.values()) <= 160


def test_기존_자연딜은_그대로다():
    """ExploringStarts를 넣다가 NaturalDeal을 건드리면 W1~W3 테스트가 조용히 깨진다."""
    assert NaturalDeal().sample(np.random.default_rng(0)) is None


def test_cards_for_key는_범위_밖_카드를_None으로_막는다():
    # 왜: 도달 불가능한 키를 실수로 넣으면 카드값 0이나 11이 나온다. 조용히 통과시키면 안 된다.
    assert cards_for_key(StateKey(4, 0, 6, 0, 0, 0)) is None
