"""이 파일은 기본전략표 36x10 격자 정의가 맞는지 검증한다
입력: blackjack_rl.chartspec 모듈의 상수와 ChartRow
출력: pytest 통과/실패
"""

import numpy as np

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ROW_INDEX, ChartTable
from blackjack_rl.state import StateKey


def test_격자는_36행_10열_360칸이다():
    assert len(CHART_ROWS) == 36
    assert len(DEALER_COLS) == 10
    assert len(CHART_ROWS) * len(DEALER_COLS) == 360


def test_행_종류별_개수가_17_9_10이다():
    kinds = [row.kind for row in CHART_ROWS]
    assert kinds.count("hard") == 17
    assert kinds.count("soft") == 9
    assert kinds.count("pair") == 10


def test_딜러_열은_2부터_11까지다():
    assert DEALER_COLS == (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def test_라벨은_모두_유일하고_ROW_INDEX와_일치한다():
    labels = [row.label for row in CHART_ROWS]
    assert len(set(labels)) == 36
    for i, label in enumerate(labels):
        assert ROW_INDEX[label] == i


def test_하드_행은_5부터_21까지다():
    hard_labels = [row.label for row in CHART_ROWS if row.kind == "hard"]
    assert hard_labels == [str(t) for t in range(5, 22)]


def test_소프트_행은_A2부터_A10까지다():
    soft_labels = [row.label for row in CHART_ROWS if row.kind == "soft"]
    assert soft_labels == [f"A,{n}" for n in range(2, 11)]


def test_페어_행은_AA와_22부터_1010까지다():
    pair_labels = [row.label for row in CHART_ROWS if row.kind == "pair"]
    assert pair_labels == ["A,A"] + [f"{r},{r}" for r in range(2, 11)]


def test_하드16_행의_키():
    row = CHART_ROWS[ROW_INDEX["16"]]
    assert row.to_key(10, can_double=True) == StateKey(16, 0, 10, 1, 0)
    assert row.to_key(10, can_double=False) == StateKey(16, 0, 10, 0, 0)


def test_소프트_A7_행의_키():
    row = CHART_ROWS[ROW_INDEX["A,7"]]
    assert row.to_key(6, can_double=True) == StateKey(18, 1, 6, 1, 0)


def test_페어_88_행의_키():
    row = CHART_ROWS[ROW_INDEX["8,8"]]
    assert row.to_key(10, can_double=True) == StateKey(16, 0, 10, 1, 1)


def test_페어_AA_행의_키():
    row = CHART_ROWS[ROW_INDEX["A,A"]]
    assert row.to_key(6, can_double=True) == StateKey(12, 1, 6, 1, 1)


def test_모든_행의_split_depth는_0이다():
    for row in CHART_ROWS:
        assert row.to_key(7, can_double=True).split_depth == 0


def test_ChartTable은_36x10_배열_두_장과_지문을_담는다():
    table = ChartTable(
        action=np.zeros((36, 10), dtype=np.int8),
        notation=np.full((36, 10), "S", dtype="<U2"),
        rules_fp="abc123def456",
    )
    assert table.action.shape == (36, 10)
    assert table.action.dtype == np.int8
    assert table.notation.shape == (36, 10)
    assert table.rules_fp == "abc123def456"
