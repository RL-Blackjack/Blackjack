"""이 파일은 정답을 본 모델과 못 본 모델을 같은 그림에 올린다.
입력: reports/supervised_comparison.json.
출력: 기대값 비교와 일반화 비교 두 그림.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

분할이름 = {"random": "무작위 분할", "structural": "구조적 분할"}


def summary_text(report: dict | None) -> str:
    """보고서가 없을 때도 화면이 죽지 않게 안내 문구를 만든다."""
    if report is None:
        return ("아직 지도학습 비교 결과가 없다. "
                "`python scripts/train_supervised.py` 를 먼저 돌려라.")
    return f"시드 {len(report['seeds'])}개 평균. DP 최적 EV {report['dp_ev'] * 100:+.4f}%"


def build_ev_chart(aggregates: list[dict], baselines: list[dict]) -> go.Figure:
    """지도학습 조건별 기대값을 막대로, 기준선을 가로선으로 그린다."""
    fig = go.Figure()
    for sk in ("random", "structural"):
        대상 = [m for m in aggregates if m["split_kind"] == sk]
        fig.add_trace(go.Bar(
            name=분할이름[sk],
            x=[f"{m['kind']} {m['ratio']:.0%}" for m in 대상],
            y=[m["ev_mean"] * 100 for m in 대상],
            error_y=dict(type="data", array=[m["ev_std"] * 100 for m in 대상]),
        ))

    선 = []
    for b in baselines:
        이름 = "DP 최적" if b["name"] == "dp_optimal" else b["name"]
        색 = "#000000" if b["family"] == "dp" else "#c62828"
        선.append(dict(type="line", xref="paper", yref="y",
                      x0=0, x1=1, y0=b["exact_ev"] * 100, y1=b["exact_ev"] * 100,
                      line=dict(color=색, width=2, dash="dash")))
        fig.add_annotation(xref="paper", x=1.0, y=b["exact_ev"] * 100,
                           text=이름, showarrow=False, xanchor="left",
                           font=dict(color=색, size=11))
    fig.update_layout(shapes=선, barmode="group", height=520,
                      yaxis_title="정확 기대값 (%)",
                      margin=dict(l=60, r=110, t=30, b=40))
    return fig


def build_generalization_chart(aggregates: list[dict]) -> go.Figure:
    """정답 비율에 따른 시험 정확도를 두 분할로 나란히 그린다."""
    fig = go.Figure()
    for sk in ("random", "structural"):
        대상 = sorted([m for m in aggregates if m["split_kind"] == sk],
                    key=lambda m: m["ratio"])
        fig.add_trace(go.Scatter(
            name=분할이름[sk], mode="lines+markers",
            x=[m["ratio"] for m in 대상],
            y=[m["test_acc_mean"] for m in 대상],
            error_y=dict(type="data", array=[m["test_acc_std"] for m in 대상]),
        ))
    fig.update_xaxes(title="정답을 보여준 비율", tickformat=".0%")
    fig.update_yaxes(title="시험 칸 정확도")
    fig.update_layout(height=460, margin=dict(l=60, r=20, t=30, b=40))
    return fig


def render(report: dict | None, dp) -> None:
    """모델 비교 탭을 그린다."""
    st.caption(summary_text(report))
    if report is None:
        return

    st.subheader("정답을 본 모델 vs 못 본 모델")
    st.plotly_chart(build_ev_chart(report["aggregates"], report["baselines"]),
                    use_container_width=True, key="tab4_ev")
    st.markdown(
        "**막대가 낮을수록 나쁘다.** 정답을 60% 보고 배운 모델이, 정답을 한 칸도 "
        "못 본 강화학습보다 아래에 있다는 것이 이 그림의 내용이다. 지도학습이 "
        "강화학습을 이기는 경우는 정답을 100% 준 랜덤포레스트뿐이고, 그때는 "
        "시드 편차가 0이다 — 학습이 아니라 표를 복사한 것이다.")

    st.subheader("외운 것인가 이해한 것인가")
    st.plotly_chart(build_generalization_chart(report["aggregates"]),
                    use_container_width=True, key="tab4_gen")
    st.markdown(
        "두 선의 간격이 곧 답이다. 무작위 분할은 이웃 칸을 보고 찍을 수 있어 쉽고, "
        "구조적 분할은 딜러 열을 통째로 빼서 본 적 없는 국면을 추론해야 한다. "
        "간격이 클수록 그 모델은 보간만 한 것이다.")
