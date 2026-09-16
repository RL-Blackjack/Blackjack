"""이 파일은 훈련/시험 분할 두 방식이 정확히 나누는지 확인한다.
입력: REACHABLE_KEYS 610개.
출력: 겹침 없음·비율·열 단위 분리에 대한 pytest 결과.
"""

import numpy as np
import pytest

from blackjack_rl.chartspec import DEALER_COLS
from blackjack_rl.models.splits import (
    RATIOS,
    SPLIT_KINDS,
    Split,
    all_splits,
    make_split,
    random_split,
    structural_split,
)
from blackjack_rl.state import REACHABLE_KEYS

키들 = tuple(REACHABLE_KEYS)


def test_상수가_설계서와_같다():
    assert RATIOS == (0.2, 0.6, 1.0)
    assert SPLIT_KINDS == ("random", "structural")


def test_딜러_열마다_상태가_정확히_61개다():
    # 왜: 이 균등성 덕분에 열 단위로 갈라도 비율이 정확히 맞는다.
    #     깨지면 구조적 분할의 ratio가 의미를 잃는다.
    센값 = {up: sum(1 for k in 키들 if k.dealer_up == up) for up in DEALER_COLS}
    assert set(센값.values()) == {61}
    assert sum(센값.values()) == 610


def test_무작위_분할은_겹치지_않고_전체를_덮는다():
    s = random_split(키들, 0.6, seed=0)
    assert len(set(s.train_idx) & set(s.test_idx)) == 0
    assert len(set(s.train_idx) | set(s.test_idx)) == 610
    assert len(s.train_idx) == int(610 * 0.6)


def test_구조적_분할은_딜러_열을_통째로_가른다():
    s = structural_split(키들, 0.6, seed=0)
    훈련열 = {키들[i].dealer_up for i in s.train_idx}
    시험열 = {키들[i].dealer_up for i in s.test_idx}
    # 왜: 한 열이 양쪽에 걸치면 '본 적 없는 딜러 카드'라는 전제가 깨진다.
    assert 훈련열 & 시험열 == set()
    assert 훈련열 | 시험열 == set(DEALER_COLS)
    assert len(훈련열) == 6
    assert len(s.train_idx) == 6 * 61


def test_구조적_분할의_빼둔_열이_기록된다():
    s = structural_split(키들, 0.6, seed=0)
    assert len(s.held_out_dealers) == 4
    시험열 = {키들[i].dealer_up for i in s.test_idx}
    assert set(s.held_out_dealers) == 시험열


def test_무작위_분할은_열을_빼두지_않는다():
    s = random_split(키들, 0.6, seed=0)
    assert s.held_out_dealers == ()


def test_비율_1이면_훈련과_시험이_모두_전체다():
    for kind in SPLIT_KINDS:
        s = make_split(kind, 키들, 1.0, seed=0)
        assert len(s.train_idx) == 610
        assert len(s.test_idx) == 610
        assert s.held_out_dealers == ()


def test_같은_시드면_같은_분할이_나온다():
    for kind in SPLIT_KINDS:
        a = make_split(kind, 키들, 0.6, seed=7)
        b = make_split(kind, 키들, 0.6, seed=7)
        assert np.array_equal(a.train_idx, b.train_idx)


def test_다른_시드면_다른_분할이_나온다():
    a = make_split("structural", 키들, 0.6, seed=0)
    b = make_split("structural", 키들, 0.6, seed=1)
    assert set(a.held_out_dealers) != set(b.held_out_dealers)


def test_비율_0_2의_구조적_분할은_두_열만_훈련한다():
    s = structural_split(키들, 0.2, seed=0)
    훈련열 = {키들[i].dealer_up for i in s.train_idx}
    assert len(훈련열) == 2
    assert len(s.train_idx) == 2 * 61


def test_전체_분할이_여섯_개이고_이름이_다르다():
    묶음 = all_splits(키들, seed=0)
    assert len(묶음) == 6
    assert len({s.name() for s in 묶음}) == 6


def test_설명이_한국어_한_줄이다():
    s = structural_split(키들, 0.6, seed=0)
    문장 = s.describe_ko()
    assert "\n" not in 문장
    assert "딜러" in 문장
    assert len(문장) > 10


def test_모르는_방식은_예외():
    with pytest.raises(ValueError):
        make_split("holdout", 키들, 0.6, seed=0)


def test_잘못된_비율은_예외():
    with pytest.raises(ValueError):
        random_split(키들, 0.0, seed=0)
    with pytest.raises(ValueError):
        random_split(키들, 1.5, seed=0)
