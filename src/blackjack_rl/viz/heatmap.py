"""이 파일은 36x10 전략표를 matplotlib 그림 한 장으로 그리는 일을 한다
입력: ChartTable(글자), margin 격자(색), 참조표 오버레이와 마스크 격자들
출력: matplotlib Figure 객체 (파일로 저장하는 일은 viz/export.py가 한다)
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ChartTable
from blackjack_rl.viz.celldata import (GRID_SHAPE, MARGIN_VMAX, mismatch_grid,
                                       shade_grid)

# 왜: matplotlib 기본 폰트에는 한글 글리프가 없어 제목이 네모(두부)로 깨진다.
#     윈도우에 항상 있는 맑은 고딕을 1순위로 두고 없으면 다음 후보로 내려간다.
KOREAN_FONTS = ("Malgun Gothic", "AppleGothic", "NanumGothic")
FALLBACK_FONT = "DejaVu Sans"

UNDECIDED_COLOR = "#d9d9d6"     # 회백색 빈칸(방문 0회)
OVERLAY_COLOR = "#111111"       # 참조표 경계선
MISMATCH_COLOR = "#d62728"      # 불일치 칸 굵은 테두리
TIE_COLOR = "#7f7f7f"           # 통계적 동점 칸 점선 테두리
CMAP_NAME = "YlGnBu"

# 하드 17행 / 소프트 9행 / 페어 10행이 갈리는 가로선 위치
ZONE_EDGES: tuple[float, float] = (16.5, 25.5)

FIG_SIZE: tuple[float, float] = (6.4, 10.2)


def use_korean_font() -> str:
    """설치돼 있는 한글 폰트를 하나 골라 rcParams에 세우고 그 이름을 돌려준다."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = FALLBACK_FONT
    for name in KOREAN_FONTS:
        if name in installed:
            chosen = name
            break
    # 왜: 없는 폰트 이름을 rcParams에 넣어 두면 글자를 그릴 때마다
    #     "findfont: Font family not found" 경고가 줄줄이 찍힌다. 있는 것만 넣는다.
    matplotlib.rcParams["font.family"] = [chosen]
    # 왜: 마이너스 기호를 유니코드 U+2212로 그리면 한글 폰트에 글리프가 없어 깨진다.
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


def _text_color(value: float) -> str:
    """배경이 진하면 흰 글자, 연하면 검은 글자."""
    if np.isnan(value):
        return "#555555"
    return "white" if value > MARGIN_VMAX * 0.55 else "#111111"


def _draw_labels(ax) -> None:
    """딜러 업카드 열 머리글과 36개 행 라벨을 축에 붙인다."""
    ax.set_xticks(range(len(DEALER_COLS)))
    ax.set_xticklabels(["A" if c == 11 else str(c) for c in DEALER_COLS])
    ax.set_yticks(range(len(CHART_ROWS)))
    ax.set_yticklabels([row.label for row in CHART_ROWS], fontsize=8)
    ax.set_xlabel("딜러 업카드")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()


def _draw_cell_grid(ax) -> None:
    """칸 사이 흰 선과 하드/소프트/페어 구역 경계선을 그린다."""
    ax.set_xticks(np.arange(-0.5, len(DEALER_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CHART_ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    for y in ZONE_EDGES:
        ax.axhline(y, color="#333333", linewidth=1.6)


def _draw_overlay_boundaries(ax, notation: np.ndarray) -> None:
    """참조표에서 표기가 달라지는 경계에만 굵은 선을 긋는다."""
    n_rows, n_cols = GRID_SHAPE
    for i in range(n_rows):
        for j in range(n_cols - 1):
            if notation[i, j] != notation[i, j + 1]:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5],
                        color=OVERLAY_COLOR, linewidth=2.2, solid_capstyle="butt")
    for i in range(n_rows - 1):
        for j in range(n_cols):
            if notation[i, j] != notation[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5],
                        color=OVERLAY_COLOR, linewidth=2.2, solid_capstyle="butt")


def _draw_box(ax, i: int, j: int, color: str, width: float, dashed: bool) -> None:
    """칸 하나에 테두리 사각형을 얹는다."""
    style = (0, (2, 2)) if dashed else "solid"
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                           edgecolor=color, linewidth=width, linestyle=style))


def policy_heatmap(chart: ChartTable, margin: np.ndarray, *,
                   overlay: ChartTable | None = None,
                   tie_mask: np.ndarray | None = None,
                   undecided_mask: np.ndarray | None = None,
                   title: str = "") -> Figure:
    """색은 margin(확신도), 글자는 S/H/D/Ds/Y/N. 미결정 칸은 글자를 찍지 않는다."""
    use_korean_font()
    shade = shade_grid(margin, undecided_mask)

    # 왜: plt.subplots 대신 Figure를 직접 만든다. pyplot은 만든 그림을 전역 목록에
    #     붙들고 있어서 GIF용으로 수십 장을 만들면 메모리 경고가 뜨고, 닫는 일을
    #     호출자가 떠안는다. Figure만 쓰면 다 쓰면 그냥 사라진다.
    fig = Figure(figsize=FIG_SIZE, layout="constrained")
    ax = fig.add_subplot()
    cmap = matplotlib.colormaps[CMAP_NAME].with_extremes(bad=UNDECIDED_COLOR)
    im = ax.imshow(shade, cmap=cmap, vmin=0.0, vmax=MARGIN_VMAX, aspect="auto")

    for i in range(GRID_SHAPE[0]):
        for j in range(GRID_SHAPE[1]):
            if undecided_mask is not None and bool(undecided_mask[i, j]):
                continue     # 왜: 방문 0회 칸은 글자를 찍지 않는다(깜빡임·분모 오염 방지)
            text = str(chart.notation[i, j])
            if text == "":
                continue
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=8, color=_text_color(float(shade[i, j])))

    _draw_cell_grid(ax)
    _draw_labels(ax)

    if tie_mask is not None:
        for i, j in zip(*np.nonzero(np.asarray(tie_mask, dtype=bool))):
            _draw_box(ax, int(i), int(j), TIE_COLOR, 1.2, True)

    if overlay is not None:
        _draw_overlay_boundaries(ax, overlay.notation)
        bad = mismatch_grid(chart.notation, overlay.notation, undecided_mask)
        for i, j in zip(*np.nonzero(bad)):
            _draw_box(ax, int(i), int(j), MISMATCH_COLOR, 2.6, False)

    bar = fig.colorbar(im, ax=ax, shrink=0.35, pad=0.02, extend="max")
    bar.set_label("확신도 = Q(최선) - Q(차선)", fontsize=8)
    if title:
        ax.set_title(title, fontsize=11, pad=24)
    return fig
