"""이 파일은 AI가 특정 상황에서 왜 그 행동을 골랐는지 보여준다.
입력: 학습 스냅샷의 최종 Q·방문수와 DP 해.
출력: 학습값과 참값을 나란히 그린 막대그래프 화면.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from blackjack_rl.dp.exact import action_values
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import ACTION_NAMES_KO, StateKey, legal_actions


def build_bars(q_row: np.ndarray, dp_row: np.ndarray, mask: np.ndarray,
               chosen: int) -> go.Figure:
    """합법 행동만 골라 학습값과 참값을 나란히 그린다."""
    자리 = [i for i in range(4) if bool(mask[i])]
    이름 = [ACTION_NAMES_KO[i] for i in 자리]

    fig = go.Figure()
    # 왜 두 계열을 겹쳐 그리는가: 학습된 Q 혼자서는 이 숫자가 맞는지 알 수 없다.
    #   옆에 분산 0의 참값을 두면 학습이 정답에 얼마나 붙었는지 한눈에 보인다.
    fig.add_trace(go.Bar(x=이름, y=[float(q_row[i]) for i in 자리],
                         name="학습된 Q", marker_color="#1565c0"))
    fig.add_trace(go.Bar(x=이름, y=[float(dp_row[i]) for i in 자리],
                         name="DP 참값", marker_color="#c62828", opacity=0.55))

    fig.update_layout(
        barmode="group", height=420,
        title=f"AI의 선택: {ACTION_NAMES_KO[chosen]}",
        yaxis_title="기대값 (베팅 1단위 기준)",
        margin=dict(l=60, r=20, t=50, b=40))
    return fig


def render(art, dp) -> None:
    """상태를 골라 그 칸의 판단 근거를 본다."""
    왼, 가운데, 오른 = st.columns(3)
    합계 = 왼.number_input("내 합계", min_value=4, max_value=21, value=16)
    업카드 = 가운데.selectbox("딜러 업카드", list(range(2, 12)),
                          index=8, format_func=lambda d: "A" if d == 11 else str(d))
    소프트 = 오른.checkbox("소프트 핸드(에이스를 11로)", value=False)
    페어 = 오른.checkbox("페어(스플릿 가능)", value=False)

    키 = StateKey(int(합계), int(소프트), int(업카드), 1, int(페어), 0)
    마스크 = legal_actions(키)

    자리 = (키.total,키.is_soft, 키.dealer_up, 키.can_double, 키.can_split, 키.split_depth)
    q_row = art.q_final[자리]
    방문 = art.n_final[자리]
    dp_row = action_values(키, RULES_V1)

    if not np.isfinite(q_row[마스크]).any():
        st.warning("이 상태는 학습 중 한 번도 나오지 않았다. 다른 상태를 골라라.")
        return

    고른것 = int(np.nanargmax(np.where(마스크, q_row, np.nan)))
    st.plotly_chart(build_bars(q_row, dp_row, 마스크, 고른것),
                    use_container_width=True, key="tab3_bars")

    a, b, c = st.columns(3)
    a.metric("DP 정답", ACTION_NAMES_KO[int(np.nanargmax(dp_row))])
    b.metric("AI 선택", ACTION_NAMES_KO[고른것])
    c.metric("이 칸 방문 횟수", f"{int(방문[마스크].sum()):,}")
