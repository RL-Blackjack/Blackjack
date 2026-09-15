"""이 파일은 난수 스트림 4개가 재현 가능하고 서로 독립인지 확인한다.
입력: blackjack_rl.rng의 make_streams와 Streams.
출력: 재현성·독립성·"탐험을 써도 카드가 안 바뀜"에 대한 pytest 결과.
"""

import numpy as np
import pytest

from blackjack_rl.rng import STREAM_NAMES, Streams, make_streams


def test_스트림이_네_개이고_이름이_정해져_있다():
    s = make_streams(0)
    assert isinstance(s, Streams)
    assert len(s) == 4
    assert STREAM_NAMES == ("deal", "explore", "init", "eval")
    assert Streams._fields == STREAM_NAMES
    for 스트림 in s:
        assert isinstance(스트림, np.random.Generator)


def test_같은_시드면_같은_수열이_나온다():
    a = make_streams(42).deal.integers(1, 11, size=1000)
    b = make_streams(42).deal.integers(1, 11, size=1000)
    assert np.array_equal(a, b)


def test_다른_시드면_다른_수열이_나온다():
    a = make_streams(42).deal.integers(1, 11, size=1000)
    b = make_streams(43).deal.integers(1, 11, size=1000)
    assert not np.array_equal(a, b)


def test_스트림끼리_서로_독립이다():
    s = make_streams(42)
    뽑기 = {이름: getattr(s, 이름).integers(0, 1000, size=1000) for 이름 in STREAM_NAMES}
    이름목록 = list(STREAM_NAMES)
    for i in range(len(이름목록)):
        for j in range(i + 1, len(이름목록)):
            assert not np.array_equal(뽑기[이름목록[i]], 뽑기[이름목록[j]])


def test_탐험을_아무리_써도_카드_수열이_안_바뀐다():
    # 왜: 이것이 "알고리즘을 바꿔도 같은 카드로 비교했다"의 코드적 증거다.
    #     ε-greedy가 난수를 몇 번 쓰든 deal 스트림은 영향을 받지 않아야 한다.
    기준 = make_streams(7).deal.integers(1, 11, size=1000)

    탐험많이쓴판 = make_streams(7)
    탐험많이쓴판.explore.random(size=100_000)
    탐험많이쓴판.init.normal(size=5_000)
    탐험많이쓴판.eval.integers(0, 10, size=5_000)
    이후 = 탐험많이쓴판.deal.integers(1, 11, size=1000)

    assert np.array_equal(기준, 이후)


def test_스트림은_이름으로도_번호로도_읽힌다():
    s = make_streams(1)
    assert s.deal is s[0]
    assert s.explore is s[1]
    assert s.init is s[2]
    assert s.eval is s[3]


def test_잘못된_시드는_예외():
    with pytest.raises(ValueError):
        make_streams(-1)
    with pytest.raises(TypeError):
        make_streams(3.5)
    with pytest.raises(TypeError):
        make_streams("42")
