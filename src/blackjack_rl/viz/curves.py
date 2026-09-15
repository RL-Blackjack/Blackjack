"""이 파일은 학습곡선과 Double Q 편향 곡선을 matplotlib 그림으로 그리는 일을 한다
입력: 에피소드 수 배열(로그 간격)과 이름별 EV 곡선 배열들, DP 최적 EV 한 개
출력: matplotlib Figure 객체 (파일로 저장하는 일은 viz/export.py가 한다)
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from blackjack_rl.viz.heatmap import use_korean_font

# 왜: 무작위 정책이 -46%에서 출발하고 최종 목표가 -0.51%다. 한 패널에 담으면
#     후반 차이가 선 굵기 안에 다 뭉개진다. 위는 급락, 아래는 확대 두 패널로 나눈다.
TOP_YLIM: tuple[float, float] = (-0.50, 0.00)
BOTTOM_YLIM: tuple[float, float] = (-0.02, -0.004)

# 왜 문헌값이 아닌가: 문헌값의 규칙 전제가 우리와 다르다. 우리 규칙으로
#     dp.exact로 직접 풀면 아래 값이 나오고, 최종 평가표(scripts/evaluate.py)도
#     같은 평가기로 같은 값을 쓴다. 문헌값 -0.1554 / -0.0596을 여기 적으면
#     한 저장소 안에서 두 그림이 서로 다른 기준선을 말하게 된다.
#     eval.simulate.baseline_evs(RULES_V1)와 같은 값임을 테스트로 고정한다.
DP_OPTIMAL_EV: float = -0.005108
BASELINE_EVS: dict[str, float] = {
    "무작위(합법 균등)": -0.460235,
    "항상 스탠드": -0.160377,
    "딜러 모방": -0.056746,
}

CURVE_FIG_SIZE: tuple[float, float] = (9.0, 7.0)
BIAS_FIG_SIZE: tuple[float, float] = (7.0, 4.2)


def _check_lengths(episodes: np.ndarray, series: Mapping[str, np.ndarray]) -> None:
    """x축과 모든 곡선의 길이가 같은지 확인한다."""
    n = len(episodes)
    for name, values in series.items():
        if len(values) != n:
            raise ValueError(
                f"'{name}' 곡선 길이 {len(values)}가 episodes 길이 {n}과 다르다")


def _plot_series(ax, episodes, series: Mapping[str, np.ndarray],
                 linestyle: str, suffix: str) -> None:
    """이름별 곡선을 한 축에 전부 그린다."""
    for name, values in series.items():
        ax.plot(episodes, values, linestyle=linestyle, linewidth=1.6,
                marker="", label=f"{name}{suffix}")


def learning_curves(episodes: np.ndarray,
                    greedy: Mapping[str, np.ndarray],
                    *,
                    dp_ev: float,
                    behavior: Mapping[str, np.ndarray] | None = None,
                    baselines: Mapping[str, float] | None = None,
                    title: str = "") -> Figure:
    """학습곡선 2단 패널. 위는 -50%~0% 급락 구간, 아래는 -2%~-0.4% 확대 구간."""
    use_korean_font()
    x = np.asarray(episodes, dtype=np.float64)
    _check_lengths(x, greedy)
    if behavior is not None:
        _check_lengths(x, behavior)
    if baselines is None:
        baselines = BASELINE_EVS

    fig = Figure(figsize=CURVE_FIG_SIZE, layout="constrained")
    ax_top, ax_bot = fig.subplots(2, 1, sharex=True)

    for ax in (ax_top, ax_bot):
        # 왜: x를 로그로 깔아야 1e3~1e5 구간의 급격한 학습이 보인다.
        #     선형 축이면 그 구간이 그림 왼쪽 끝 1mm 안에 전부 들어간다.
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
        ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
        _plot_series(ax, x, greedy, "-", " (그리디)")
        if behavior is not None:
            _plot_series(ax, x, behavior, "--", " (행동, ε 포함)")
        ax.axhline(dp_ev, color="black", linestyle=":", linewidth=1.8,
                   label=f"DP 최적 EV {dp_ev * 100:.4f}%")

    for name, value in baselines.items():
        ax_top.axhline(value, color="#999999", linestyle="-.", linewidth=1.0)
        ax_top.annotate(f"{name} {value * 100:.2f}%", (x[0], value),
                        fontsize=7, color="#666666",
                        xytext=(2, 2), textcoords="offset points")

    ax_top.set_ylim(*TOP_YLIM)
    ax_bot.set_ylim(*BOTTOM_YLIM)
    ax_bot.set_xlabel("에피소드 수 (로그)")
    ax_top.set_ylabel("정확 EV (DP 정책평가)")
    ax_bot.set_ylabel("확대")
    ax_top.legend(fontsize=7, loc="lower right", ncols=2)
    if title:
        ax_top.set_title(title, fontsize=12)
    return fig


def bias_curves(episodes: np.ndarray,
                series: Mapping[str, np.ndarray],
                *, title: str = "") -> Figure:
    """max_a Q(s,a) - V*(s) 편향 곡선. 프로브 50칸으로 잰 값을 그린다."""
    use_korean_font()
    x = np.asarray(episodes, dtype=np.float64)
    _check_lengths(x, series)

    fig = Figure(figsize=BIAS_FIG_SIZE, layout="constrained")
    ax = fig.add_subplot()
    ax.set_xscale("log")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.4)
    _plot_series(ax, x, series, "-", "")
    # 왜: 0이 '편향 없음'이다. 기준선을 깔아야 위로 떴는지 아래로 갔는지가 보인다.
    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.set_xlabel("에피소드 수 (로그)")
    ax.set_ylabel("max Q - V*  (프로브 50칸 평균)")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=12)
    return fig
