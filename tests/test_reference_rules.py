"""이 파일은 출판 참조표 CSV가 36x10이고 규칙 지문이 맞는지 검증한다
입력: src/blackjack_rl/data/basic_strategy_4d_s17_das.csv
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS
from blackjack_rl.reference import (
    REFERENCE_CSV,
    REFERENCE_RULES,
    ReferenceFormatError,
    ReferenceRuleMismatch,
    load_reference_chart,
)
from blackjack_rl.rules import RULES_V1


def test_참조표는_36행_10열이다():
    chart = load_reference_chart()
    assert chart.notation.shape == (36, 10)
    assert chart.notation.dtype == np.dtype("<U2")
    assert chart.row_labels == tuple(row.label for row in CHART_ROWS)
    assert chart.dealer_cols == tuple(DEALER_COLS)


def test_규칙_지문이_다르면_로딩_자체가_예외다():
    # 왜: 무한덱 RULES_V1으로 4덱 참조표를 읽으면 규칙 차이와 학습 부족이 섞인다.
    with pytest.raises(ReferenceRuleMismatch):
        load_reference_chart(RULES_V1)


def test_참조_규칙은_덱_모형만_빼고_정본과_같다():
    assert REFERENCE_RULES.deck_mode == "shoe"
    assert REFERENCE_RULES.decks == 4
    assert REFERENCE_RULES.dealer_hits_soft_17 == RULES_V1.dealer_hits_soft_17
    assert REFERENCE_RULES.double_after_split == RULES_V1.double_after_split
    assert REFERENCE_RULES.max_split_depth == RULES_V1.max_split_depth
    assert REFERENCE_RULES.blackjack_payout == RULES_V1.blackjack_payout
    assert REFERENCE_RULES.fingerprint() != RULES_V1.fingerprint()


def test_페어_행은_Y_또는_N만_쓴다():
    chart = load_reference_chart()
    for index, row in enumerate(CHART_ROWS):
        cells = set(chart.notation[index].tolist())
        if row.kind == "pair":
            assert cells <= {"Y", "N"}, f"{row.label}: {cells}"
        else:
            assert cells <= {"H", "S", "D", "Ds"}, f"{row.label}: {cells}"


def test_알려진_대표_칸의_값이_출판표와_같다():
    chart = load_reference_chart()
    label_to_index = {label: i for i, label in enumerate(chart.row_labels)}
    col = {up: i for i, up in enumerate(chart.dealer_cols)}

    def cell(label: str, dealer: int) -> str:
        return str(chart.notation[label_to_index[label], col[dealer]])

    assert cell("16", 10) == "H"      # 서렌더 없음 -> 16 vs 10은 히트
    assert cell("11", 11) == "H"      # S17 멀티덱 -> 11 vs A는 더블이 아니다
    assert cell("12", 2) == "H"
    assert cell("12", 4) == "S"
    assert cell("A,7", 3) == "Ds"
    assert cell("A,7", 9) == "H"
    assert cell("8,8", 11) == "Y"
    assert cell("9,9", 7) == "N"
    assert cell("5,5", 6) == "N"
    assert cell("4,4", 5) == "Y"      # DAS이므로 4,4는 5,6에서 스플릿


def test_지문_줄이_없으면_형식_예외다(tmp_path):
    원본 = REFERENCE_CSV.read_text(encoding="utf-8")
    깨진표 = tmp_path / "broken.csv"
    깨진표.write_text(
        "\n".join(line for line in 원본.splitlines()
                  if not line.startswith("# rules_fingerprint=")),
        encoding="utf-8")
    with pytest.raises(ReferenceFormatError):
        load_reference_chart(REFERENCE_RULES, path=깨진표)


def test_행이_하나_빠지면_형식_예외다(tmp_path):
    줄들 = REFERENCE_CSV.read_text(encoding="utf-8").splitlines()
    깨진표 = tmp_path / "short.csv"
    깨진표.write_text("\n".join(줄들[:-1]), encoding="utf-8")
    with pytest.raises(ReferenceFormatError):
        load_reference_chart(REFERENCE_RULES, path=깨진표)
