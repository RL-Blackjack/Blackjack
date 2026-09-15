"""이 파일은 출판된 기본전략 참조표 CSV를 읽어 36x10 격자로 돌려준다
입력: data/basic_strategy_4d_s17_das.csv 파일과 대조할 RuleSet
출력: ReferenceChart(표기 36x10, 행 라벨 36개, 딜러 열 10개, 규칙 지문)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .chartspec import CHART_ROWS, DEALER_COLS
from .rules import RULES_V1, RuleSet

# 왜: 출판표는 4덱 슈에서 계산된 유한덱 해다. 정본 RULES_V1(무한덱)과 플레이 규칙은
#     전부 같고 덱 모형 하나만 다르다. 그 유일한 차이에 이름을 붙여 한 곳에 못박는다.
REFERENCE_RULES: RuleSet = replace(RULES_V1, deck_mode="shoe", decks=4)

# 왜: DP 갭이 이보다 작으면 통계적 동점으로 본다. W6의 eval/agreement.py가 이 값을
#     그대로 다시 쓴다. 상수가 두 곳에 생기면 숫자가 조용히 갈라진다.
TIE_DELTA = 0.005

REFERENCE_CSV = Path(__file__).parent / "data" / "basic_strategy_4d_s17_das.csv"

NON_PAIR_NOTATIONS = ("H", "S", "D", "Ds")
PAIR_NOTATIONS = ("Y", "N")


class ReferenceRuleMismatch(Exception):
    """참조표의 규칙 지문이 현재 RuleSet과 다를 때 올린다."""


class ReferenceFormatError(Exception):
    """참조표 CSV의 모양이 36x10이 아니거나 표기가 낯설 때 올린다."""


@dataclass(frozen=True)
class ReferenceChart:
    notation: np.ndarray            # '<U2'[36,10]
    row_labels: tuple[str, ...]     # CHART_ROWS와 같은 순서
    dealer_cols: tuple[int, ...]    # DEALER_COLS와 같은 순서
    rules_fp: str
    source: str


def _split_meta_and_body(text: str) -> tuple[dict[str, str], list[list[str]]]:
    """'#'로 시작하는 줄은 메타로, 나머지는 CSV 본문으로 가른다."""
    meta: dict[str, str] = {}
    data_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            item = stripped.lstrip("#").strip()
            if "=" in item:
                key, value = item.split("=", 1)
                meta[key.strip()] = value.strip()
            continue
        data_lines.append(stripped)
    # 왜: 행 라벨 "A,2"와 "8,8" 안에 쉼표가 있어서 큰따옴표로 감쌌다.
    #     naive split(",")로는 못 읽고 표준 csv 모듈이 필요하다.
    body = [[cell.strip() for cell in row] for row in csv.reader(data_lines)]
    return meta, body


def _check_shape(body: list[list[str]]) -> None:
    """헤더 1줄 + 데이터 36줄, 각 줄은 라벨 1칸 + 딜러 10칸인지 확인한다."""
    if len(body) != 1 + len(CHART_ROWS):
        raise ReferenceFormatError(
            f"데이터 줄 수가 {len(body) - 1}개다. {len(CHART_ROWS)}개여야 한다.")
    for row in body:
        if len(row) != 1 + len(DEALER_COLS):
            raise ReferenceFormatError(
                f"'{row[0]}' 줄의 칸 수가 {len(row) - 1}개다. "
                f"{len(DEALER_COLS)}개여야 한다.")


def _check_header(header: list[str]) -> None:
    """딜러 열 머리글이 DEALER_COLS와 같은 순서인지 확인한다."""
    try:
        cols = tuple(int(cell) for cell in header[1:])
    except ValueError as err:
        raise ReferenceFormatError(f"딜러 열 머리글이 숫자가 아니다: {header[1:]}") from err
    if cols != tuple(DEALER_COLS):
        raise ReferenceFormatError(f"딜러 열이 {cols}다. {tuple(DEALER_COLS)}여야 한다.")


def _check_labels_and_cells(rows: list[list[str]]) -> None:
    """행 라벨 순서와 각 칸의 표기가 규칙에 맞는지 확인한다."""
    for chart_row, row in zip(CHART_ROWS, rows):
        if row[0] != chart_row.label:
            raise ReferenceFormatError(
                f"행 라벨이 '{row[0]}'다. '{chart_row.label}'이어야 한다.")
        allowed = PAIR_NOTATIONS if chart_row.kind == "pair" else NON_PAIR_NOTATIONS
        for cell in row[1:]:
            if cell not in allowed:
                raise ReferenceFormatError(
                    f"'{chart_row.label}' 행의 표기 '{cell}'는 {allowed} 중에 없다.")


def load_reference_chart(rules: RuleSet = REFERENCE_RULES,
                         path: Path | None = None) -> ReferenceChart:
    """참조표를 읽는다. 규칙 지문이 다르면 경고가 아니라 예외로 멈춘다."""
    csv_path = REFERENCE_CSV if path is None else path
    meta, body = _split_meta_and_body(csv_path.read_text(encoding="utf-8"))

    if "rules_fingerprint" not in meta:
        raise ReferenceFormatError(f"{csv_path}에 rules_fingerprint 메타 줄이 없다.")
    stored = meta["rules_fingerprint"]
    current = rules.fingerprint()
    if stored != current:
        # 왜: 규칙이 어긋난 채 학습하면 나중에 '규칙 차이'와 '학습 부족'을 구분할 수 없다.
        raise ReferenceRuleMismatch(
            f"참조표 규칙 지문 {stored} != 현재 규칙 지문 {current}. "
            f"CSV: {csv_path}")

    _check_shape(body)
    _check_header(body[0])
    _check_labels_and_cells(body[1:])

    notation = np.array([row[1:] for row in body[1:]], dtype="<U2")
    return ReferenceChart(
        notation=notation,
        row_labels=tuple(row.label for row in CHART_ROWS),
        dealer_cols=tuple(DEALER_COLS),
        rules_fp=stored,
        source=meta.get("source", "출처 미기재"),
    )
