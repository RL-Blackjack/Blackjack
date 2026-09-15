"""이 파일은 전략표 36x10 칸을 색칠하고 표시할 마스크 배열의 계산을 한다
입력: margin 격자 float[36,10], 방문수 격자 int[36,10], 표기 격자 '<U2'[36,10]
출력: undecided / tie / mismatch / shade 격자 (matplotlib 없이 numpy만 쓴다)
"""

from __future__ import annotations

import numpy as np

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS
from blackjack_rl.eval.agreement import TRIVIAL_DELTA
from blackjack_rl.reference import TIE_DELTA

GRID_SHAPE: tuple[int, int] = (len(CHART_ROWS), len(DEALER_COLS))

# 왜 상수를 새로 안 만드는가: 색의 최대값과 '자명한 칸' 임계값은 같은 숫자여야
#     그림과 일치율 표가 같은 기준을 말한다. 0.20을 두 곳에 따로 적으면
#     한쪽만 고쳤을 때 아무도 모른다. eval.agreement의 정의를 그대로 쓴다.
MARGIN_VMAX: float = TRIVIAL_DELTA

# 왜 margin 계산 함수가 여기 없는가: eval.agreement.chart_margin 하나뿐이다.
#     train/runner도 그 함수로 npz의 chart_margin을 채우므로, 화면에 칠하는 색과
#     저장된 배열이 같은 정의에서 나온다. 여기서 다시 만들면 두 벌이 된다.


def undecided_grid(visits: np.ndarray) -> np.ndarray:
    """방문수가 0인 칸(아직 결정되지 않은 칸)을 True로 표시한다."""
    arr = np.asarray(visits)
    if arr.shape != GRID_SHAPE:
        raise ValueError(f"visits 모양은 {GRID_SHAPE}여야 한다: {arr.shape}")
    return arr == 0


def tie_grid(margin: np.ndarray, delta: float = TIE_DELTA) -> np.ndarray:
    """margin이 delta보다 작은 칸(통계적 동점)을 True로 표시한다."""
    arr = np.asarray(margin, dtype=np.float64)
    if arr.shape != GRID_SHAPE:
        raise ValueError(f"margin 모양은 {GRID_SHAPE}여야 한다: {arr.shape}")
    # 왜: nan은 비교가 항상 False라 자동으로 빠지지만 경고가 뜬다. 미리 채운다.
    filled = np.where(np.isnan(arr), np.inf, arr)
    return filled < delta


def mismatch_grid(ai_notation: np.ndarray, ref_notation: np.ndarray,
                  undecided: np.ndarray | None = None) -> np.ndarray:
    """AI 표기와 참조표 표기가 다른 칸을 True로 표시한다. 미결정 칸은 제외한다."""
    a = np.asarray(ai_notation)
    b = np.asarray(ref_notation)
    if a.shape != GRID_SHAPE or b.shape != GRID_SHAPE:
        raise ValueError(
            f"표기 격자 모양은 둘 다 {GRID_SHAPE}여야 한다: {a.shape}, {b.shape}")
    diff = a != b
    if undecided is not None:
        # 왜: 글자를 찍지 않은 칸을 불일치로 세면 일치율 분모가 오염된다.
        diff = diff & ~np.asarray(undecided, dtype=bool)
    return diff


def shade_grid(margin: np.ndarray, undecided: np.ndarray | None = None) -> np.ndarray:
    """색으로 쓸 배열. 0~MARGIN_VMAX로 자르고 미결정 칸은 nan으로 비운다."""
    arr = np.asarray(margin, dtype=np.float64).copy()
    arr = np.clip(arr, 0.0, MARGIN_VMAX)
    if undecided is not None:
        arr[np.asarray(undecided, dtype=bool)] = np.nan
    return arr
