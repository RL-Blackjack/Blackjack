"""이 파일은 전략표가 학습과 함께 완성되는 과정을 그린다.
입력: 학습 스냅샷(프레임별 표기·확신도·방문수)과 DP 해.
출력: Plotly 히트맵과 정답표 오버레이가 붙은 화면.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS
from blackjack_rl.eval.agreement import load_reference_chart

ACTION_COLORS: dict[str, str] = {
    "S": "#c62828",   # 스탠드 - 빨강
    "H": "#1565c0",   # 히트 - 파랑
    "D": "#2e7d32",   # 더블 - 초록
    "Ds": "#2e7d32",
    "Y": "#6a1b9a",   # 스플릿 - 보라
    "N": "#616161",   # 스플릿 안 함 - 회색
}

행이름 = [r.label for r in CHART_ROWS]
열이름 = ["A" if d == 11 else str(d) for d in DEALER_COLS]


def build_heatmap(chart_action: np.ndarray, chart_notation: np.ndarray,
                  margin: np.ndarray, undecided: np.ndarray) -> go.Figure:
    """확신도로 색을 칠하고 행동은 글자로만 찍는다."""
    # 왜 색을 margin으로 칠하는가: 행동으로 칠하면 프레임이 넘어갈 때 색이
    #   딱딱 바뀐다. 최선과 차선의 가치 차이로 칠하면 색이 서서히 진해지며
    #   'AI의 확신이 굳어가는' 장면이 된다.
    z = np.where(undecided, np.nan, margin.astype(float))
    글자 = np.where(undecided, "", chart_notation)

    fig = go.Figure(go.Heatmap(
        z=z, text=글자, texttemplate="%{text}",
        x=열이름, y=행이름,
        colorscale="Blues", zmin=0.0,
        hovertemplate="%{y} vs 딜러 %{x}<br>행동 %{text}<br>확신도 %{z:.3f}<extra></extra>",
        colorbar=dict(title="확신도"),
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=760, margin=dict(l=60, r=20, t=30, b=40))
    return fig


def overlay_reference(fig: go.Figure, ref_notation: np.ndarray,
                      ai_notation: np.ndarray) -> go.Figure:
    """정답표와 다른 칸에만 굵은 테두리를 그린다."""
    테두리 = []
    for i in range(ref_notation.shape[0]):
        for j in range(ref_notation.shape[1]):
            if str(ai_notation[i, j]) == "":
                # 왜 건너뛰는가: 학습으로 채울 수 없는 칸을 '틀렸다'고 표시하면
                #   "AI가 30칸을 못 배웠다"는 오해를 만든다.
                continue
            if str(ai_notation[i, j]) != str(ref_notation[i, j]):
                테두리.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                    line=dict(color="#ff6f00", width=3),
                ))
    fig.update_layout(shapes=테두리)
    return fig


def render(art, dp) -> None:
    """슬라이더로 학습 시점을 넘기며 표가 채워지는 것을 본다."""
    프레임수 = len(art.episodes)
    자리 = st.slider("학습 에피소드", 0, 프레임수 - 1, 프레임수 - 1,
                    format="", key="tab1_frame")
    st.caption(f"에피소드 {int(art.episodes[자리]):,}")

    미결정 = art.chart_visits[자리] == 0
    fig = build_heatmap(art.chart_action[자리], art.chart_notation[자리],
                        art.chart_margin[자리], 미결정)

    참조표 = load_reference_chart()
    if st.checkbox("1956년 기본 전략표와 비교", value=False, key="tab1_overlay"):
        overlay_reference(fig, 참조표.notation, art.chart_notation[자리])

    왼쪽, 오른쪽, 가운데 = st.columns(3)
    왼쪽.metric("360칸 일치율", f"{art.agree_a[자리] * 100:.2f}%")
    오른쪽.metric("비자명 칸 일치율", f"{art.agree_b[자리] * 100:.2f}%")
    가운데.metric("미결정 칸", f"{int(미결정.sum())}칸")

    st.plotly_chart(fig, use_container_width=True, key="tab1_chart")

    if int(미결정.sum()) > 0:
        st.info(
            "미결정 칸은 학습으로 채울 수 없는 자리다. 하드 20과 A,10은 두 장으로 "
            "만들면 반드시 페어가 되어 탐험적 시작이 만들 수 없고, 하드 21은 애초에 "
            "도달 가능한 상태가 아니다. AI가 못 배운 것이 아니라 갈 수 없는 칸이다.")
