"""이 파일은 모델 비교 화면이 보고서를 올바르게 그리는지 확인한다.
입력: 가짜 비교 보고서.
출력: pytest 통과/실패.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from tabs import tab4_compare  # noqa: E402


def 가짜보고서():
    집계 = []
    for kind in ("rf", "mlp"):
        for sk in ("random", "structural"):
            for ratio in (0.2, 0.6, 1.0):
                집계.append({
                    "kind": kind, "split_kind": sk, "ratio": ratio, "n_seeds": 5,
                    "ev_mean": -0.02, "ev_std": 0.005, "gap_pp_mean": 1.5,
                    "test_acc_mean": 0.85, "test_acc_std": 0.02,
                    "tier_a_mean": 0.9, "fit_seconds_mean": 0.3,
                    "n_train_labels": 366,
                })
    기준선 = [
        {"name": "dp_optimal", "family": "dp", "exact_ev": -0.005108, "gap_pp": 0.0},
        {"name": "rl_mc", "family": "rl", "exact_ev": -0.007077, "gap_pp": 0.197},
    ]
    return {"schema_version": 1, "aggregates": 집계, "baselines": 기준선,
            "dp_ev": -0.005108, "seeds": [0, 1, 2, 3, 4]}


def test_EV_그림에_지도학습과_기준선이_모두_있다():
    r = 가짜보고서()
    fig = tab4_compare.build_ev_chart(r["aggregates"], r["baselines"])
    assert len(fig.data) >= 1
    assert len(fig.layout.shapes) >= 1      # 기준선은 가로선으로 그린다


def test_일반화_그림은_두_분할을_나란히_그린다():
    r = 가짜보고서()
    fig = tab4_compare.build_generalization_chart(r["aggregates"])
    이름들 = {t.name for t in fig.data}
    assert "무작위 분할" in 이름들
    assert "구조적 분할" in 이름들


def test_보고서가_없으면_안내만_한다():
    # 왜: 아직 격자를 안 돌린 상태에서도 대시보드가 죽지 않아야 한다.
    assert tab4_compare.summary_text(None).startswith("아직")
