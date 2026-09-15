"""이 파일은 RuleSet이 진짜로 얼어 있고 지문이 규칙마다 다른지 확인한다.
입력: blackjack_rl.rules의 RuleSet 클래스와 프리셋 4종.
출력: 불변성·지문 유일성·검증 예외·한국어 설명에 대한 pytest 결과.
"""

import dataclasses
from dataclasses import replace

import pytest

from blackjack_rl.rules import (
    RULES_H17,
    RULES_NO_DAS,
    RULES_SHOE,
    RULES_V1,
    RuleSet,
)


def test_기본값이_설계서와_같다():
    r = RuleSet()
    assert r.deck_mode == "infinite"
    assert r.decks == 4
    assert r.penetration == 0.75
    assert r.dealer_hits_soft_17 is False
    assert r.blackjack_payout == 1.5
    assert r.dealer_peeks is True
    assert r.double_any_two is True
    assert r.double_after_split is True
    assert r.max_split_depth == 3
    assert r.max_hands_per_round == 4
    assert r.split_aces_one_card is True
    assert r.resplit_aces is False
    assert r.surrender is False
    assert r.insurance is False
    assert r.include_split_depth_in_state is False


def test_같은_규칙이면_지문이_항상_같다():
    assert RuleSet().fingerprint() == RuleSet().fingerprint()
    assert RULES_V1.fingerprint() == RuleSet().fingerprint()


def test_지문은_12자_16진수다():
    fp = RULES_V1.fingerprint()
    assert len(fp) == 12
    assert all(ch in "0123456789abcdef" for ch in fp)


def test_필드가_하나라도_바뀌면_지문이_달라진다():
    변형들 = [
        {"dealer_hits_soft_17": True},
        {"dealer_peeks": False},
        {"double_any_two": False},
        {"double_after_split": False},
        {"blackjack_payout": 1.2},
        {"max_split_depth": 1},
        {"max_hands_per_round": 2},
        {"split_aces_one_card": False},
        {"resplit_aces": True},
        {"include_split_depth_in_state": True},
        {"deck_mode": "shoe"},
    ]
    지문들 = [RULES_V1.fingerprint()]
    for 바꿀값 in 변형들:
        지문들.append(replace(RULES_V1, **바꿀값).fingerprint())
    assert len(set(지문들)) == len(지문들)


def test_슈_모드에서_덱수와_페네트레이션도_지문에_들어간다():
    a = RULES_SHOE.fingerprint()
    b = replace(RULES_SHOE, decks=6).fingerprint()
    c = replace(RULES_SHOE, penetration=0.5).fingerprint()
    assert len({a, b, c}) == 3


def test_얼어있어서_수정하면_예외가_난다():
    r = RuleSet()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.decks = 8


def test_해시가_되어서_사전_키로_쓸_수_있다():
    # 왜: dp/dealer_dp.py가 @lru_cache(rules=...)로 캐싱하므로 해시 가능해야 한다.
    표 = {RULES_V1: "정본", RULES_H17: "H17"}
    assert 표[RuleSet()] == "정본"


def test_무한덱에서_덱수나_페네트레이션을_바꾸면_예외():
    with pytest.raises(ValueError):
        RuleSet(deck_mode="infinite", decks=6)
    with pytest.raises(ValueError):
        RuleSet(deck_mode="infinite", penetration=0.5)


def test_슈_모드의_잘못된_값은_예외():
    with pytest.raises(ValueError):
        RuleSet(deck_mode="shoe", decks=0)
    with pytest.raises(ValueError):
        RuleSet(deck_mode="shoe", decks=9)
    with pytest.raises(ValueError):
        RuleSet(deck_mode="shoe", penetration=0.0)
    with pytest.raises(ValueError):
        RuleSet(deck_mode="shoe", penetration=1.5)


def test_모르는_덱모드는_예외():
    with pytest.raises(ValueError):
        RuleSet(deck_mode="double_deck")


def test_배당과_스플릿_상한_검증():
    with pytest.raises(ValueError):
        RuleSet(blackjack_payout=0.5)
    with pytest.raises(ValueError):
        RuleSet(max_split_depth=4)
    with pytest.raises(ValueError):
        RuleSet(max_split_depth=-1)
    with pytest.raises(ValueError):
        RuleSet(max_hands_per_round=9)
    with pytest.raises(ValueError):
        RuleSet(max_split_depth=3, max_hands_per_round=1)


def test_범위밖_규칙은_켜지지_않는다():
    # 왜: 서렌더·인슈어런스는 설계서 §10에서 제외됐다. True로 두면 조용히 무시되어
    #     결과가 틀린 채로 나오므로 아예 만들지 못하게 막는다.
    with pytest.raises(ValueError):
        RuleSet(surrender=True)
    with pytest.raises(ValueError):
        RuleSet(insurance=True)


def test_한국어_설명이_한_문단이고_핵심어를_담는다():
    문장 = RULES_V1.describe_ko()
    assert "\n" not in 문장
    assert len(문장) > 50
    assert "무한덱" in 문장
    assert "S17" in 문장
    assert "3:2" in 문장
    assert "DAS" in 문장
    assert RULES_V1.fingerprint() in 문장


def test_규칙이_바뀌면_설명_문장도_바뀐다():
    assert "H17" in RULES_H17.describe_ko()
    assert "S17" in RULES_V1.describe_ko()
    assert "스플릿 후 더블 금지" in RULES_NO_DAS.describe_ko()
    assert "4덱" in RULES_SHOE.describe_ko()


def test_프리셋들이_설계서대로_만들어졌다():
    assert RULES_H17.dealer_hits_soft_17 is True
    assert RULES_H17.deck_mode == "infinite"
    assert RULES_NO_DAS.double_after_split is False
    assert RULES_SHOE.deck_mode == "shoe"
    assert RULES_SHOE.decks == 4
    assert RULES_SHOE.penetration == 0.75
