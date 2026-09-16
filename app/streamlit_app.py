"""이 파일은 연구 대시보드의 탭을 배치하고 재현성 정보를 아래에 붙인다.
입력: artifacts/ 의 학습 스냅샷과 reports/ 의 비교 보고서.
출력: 브라우저 화면.
"""

from __future__ import annotations

import streamlit as st

import loaders
from tabs import tab1_policy, tab2_curves, tab3_explain, tab4_compare

st.set_page_config(page_title="블랙잭 강화학습 연구", layout="wide")

st.title("블랙잭 강화학습 연구 대시보드")
st.caption("규칙만 알고 전략은 모르는 AI가 스스로 찾아낸 전략표를, "
           "출판된 기본전략표와 비교한다.")

실행목록 = loaders.list_runs()
if not 실행목록:
    st.warning("artifacts/ 에 학습 결과가 없다. "
               "`python scripts/train.py` 를 먼저 돌려라.")
    st.stop()

선택 = st.sidebar.selectbox("학습 결과", 실행목록, format_func=lambda p: p.stem)
art = loaders.load_snapshot(str(선택))
dp = loaders.load_dp()

탭1, 탭2, 탭3, 탭4 = st.tabs([
    "전략표가 완성되는 과정", "학습곡선", "AI의 판단 근거", "모델 비교",
])

with 탭1:
    tab1_policy.render(art, dp)
with 탭2:
    tab2_curves.render(art, dp)
with 탭3:
    tab3_explain.render(art, dp)
with 탭4:
    tab4_compare.render(loaders.load_comparison(), dp)

# 왜 메타를 항상 띄우는가: "이 결과 진짜냐"는 질문에 화면을 가리키며 답할 수 있어야 한다.
with st.expander("재현성 정보"):
    st.json(art.meta)
