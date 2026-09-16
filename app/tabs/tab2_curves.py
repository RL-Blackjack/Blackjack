"""이 파일은 학습이 진행되며 기대값이 올라가는 과정을 그린다.
입력: 학습 스냅샷의 에피소드·그리디 EV·행동정책 EV와 DP 해.
출력: Plotly 학습곡선 화면.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from blackjack_rl.eval.simulate import baseline_evs
from blackjack_rl.rules import RULES_V1


def build_curve(episodes: np.ndarray, ev_greedy: np.ndarray, ev_behavior: np.ndarray,
                dp_ev: float, baselines: dict[str, float]) -> go.Figure:
    """학습곡선에 DP 상한선과 기준선들을 함께 그린다."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=episodes, y=ev_greedy * 100, name="그리디 정책",
                             mode="lines", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=episodes, y=ev_behavior * 100, name="행동정책(탐험 포함)",
                             mode="lines", line=dict(width=2, dash="dot")))
    fig.add_trace(go.Scatter(
        x=[episodes[0], episodes[-1]], y=[dp_ev * 100, dp_ev * 100],
        name="DP 최적(상한)", mode="lines",
        line=dict(color="#000000", width=2, dash="dash")))

    for 이름, 값 in baselines.items():
        fig.add_trace(go.Scatter(
            x=[episodes[0], episodes[-1]], y=[값 * 100, 값 * 100],
            name=이름, mode="lines", line=dict(width=1, dash="dashdot")))

    # 왜 x축이 로그인가: 초반 변화가 압도적으로 커서 선형 축이면 후반의
    #   미세한 개선이 전부 뭉개진다. 스냅샷도 로그 간격으로 찍혀 있다.
    fig.update_xaxes(type="log", title="학습 에피소드")
    fig.update_yaxes(title="기대값 (%)")
    fig.update_layout(height=560, hovermode="x unified",
                      margin=dict(l=60, r=20, t=30, b=40))
    return fig


def render(art, dp) -> None:
    """학습곡선 탭을 그린다."""
    # 왜 baseline_evs를 부르는가: 숫자를 직접 적으면 이 그림과 최종 평가표가
    #   같은 정책에 대해 서로 다른 값을 말하게 된다.
    # 주의: baseline_evs는 dict가 아니라 list[tuple[str, float]]를 돌려주고,
    #   DP 항목의 이름은 "DP 최적(상한)"이다(2026-09-16 실제 코드로 확인).
    #   DP는 아래에서 따로 점선으로 그리므로 여기서는 뺀다.
    기준 = {이름: 값 for 이름, 값 in baseline_evs(RULES_V1)
           if not 이름.startswith("DP 최적")}

    fig = build_curve(art.episodes, art.ev_greedy, art.ev_behavior,
                      float(dp.ev_initial), 기준)
    st.plotly_chart(fig, use_container_width=True, key="tab2_chart")

    왼쪽, 오른쪽 = st.columns(2)
    왼쪽.metric("최종 그리디 EV", f"{art.ev_greedy[-1] * 100:+.4f}%")
    오른쪽.metric("DP 최적과의 격차",
                f"{(dp.ev_initial - art.ev_greedy[-1]) * 100:.4f}%p")

    st.caption(
        "무작위 플레이 기준선은 −46%다. 학습이 진행되며 DP 최적(−0.51%)에 "
        "가까워지는 것이 이 그림의 내용이다. 행동정책이 그리디보다 낮은 것은 "
        "탐험 때문이며, 보고하는 값은 언제나 그리디 쪽이다.")
