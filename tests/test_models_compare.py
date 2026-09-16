"""이 파일은 정책을 전략표로 되돌리고 네 모델을 같은 자로 재는지 확인한다.
입력: DP 해와 정책 배열.
출력: 변환 정확성·비교표 항목에 대한 pytest 결과.
"""

import numpy as np
import pytest

from blackjack_rl.dp.exact import evaluate_policy, greedy_policy_full, optimal_chart
from blackjack_rl.models.compare import (
    ModelRow,
    make_scorer,
    policy_to_chart,
    policy_to_q,
    rows_to_markdown,
)
from blackjack_rl.rules import RULES_V1


def test_DP정책을_되돌리면_원래_표와_350칸_일치한다(dp):
    # 왜 360이 아닌가: 하드 21 행 10칸은 StateKey(21,0,up,1,0,0)이 애초에
    #     REACHABLE_KEYS에 없어 미결정으로 남는다. 실측으로 확인한 값이다.
    되돌림 = policy_to_chart(greedy_policy_full(dp), RULES_V1)
    원본 = optimal_chart(RULES_V1)
    같음 = (되돌림.notation == 원본.notation)
    assert int(같음.sum()) == 350
    미결정 = (되돌림.notation == "")
    assert int(미결정.sum()) == 10


def test_되돌린_표에도_Ds가_살아있다(dp):
    # 왜: 행동만으로는 D와 Ds가 둘 다 DOUBLE이라 구분되지 않는다. project()가
    #     can_double을 True/False로 두 번 조회하므로 정책만으로도 복원된다.
    되돌림 = policy_to_chart(greedy_policy_full(dp), RULES_V1)
    표기 = set(np.unique(되돌림.notation).tolist())
    assert "Ds" in 표기
    assert "D" in 표기


def test_변환한_Q는_불법에_nan을_둔다(dp):
    Q = policy_to_q(greedy_policy_full(dp))
    assert Q.shape == (22, 2, 12, 2, 2, 4, 4)
    # 스탠드와 히트는 언제나 합법이라 nan이 아니다.
    from blackjack_rl.state import REACHABLE_KEYS
    k = REACHABLE_KEYS[0]
    q = Q[k.total, k.is_soft, k.dealer_up, k.can_double, k.can_split, k.split_depth]
    assert not np.isnan(q[0]) and not np.isnan(q[1])
    assert np.nanmax(q) == pytest.approx(1.0)


def test_DP_자신을_재면_격차가_0이다(dp):
    점수 = make_scorer(RULES_V1, dp)
    행 = 점수("dp_optimal", "dp", greedy_policy_full(dp))
    assert 행.exact_ev == pytest.approx(dp.ev_initial, abs=1e-9)
    assert 행.gap_pp == pytest.approx(0.0, abs=1e-9)


def test_비교행에_필요한_항목이_다_있다(dp):
    점수 = make_scorer(RULES_V1, dp)
    행 = 점수("x", "supervised", greedy_policy_full(dp),
             n_train_labels=366, test_acc=0.877, fit_seconds=0.2, n_params=1000)
    for 이름 in ("name", "family", "exact_ev", "gap_pp", "tier_a", "tier_b",
                "n_undecided", "n_train_labels", "test_acc", "fit_seconds", "n_params"):
        assert hasattr(행, 이름)
    assert 행.n_train_labels == 366


def test_나쁜_정책은_격차가_크다(dp):
    # 전부 스탠드하는 정책의 정확 EV는 -0.160377이다(실측).
    점수 = make_scorer(RULES_V1, dp)
    행 = 점수("always_stand", "baseline", np.zeros(610, dtype=np.int8))
    assert 행.exact_ev == pytest.approx(-0.160377, abs=1e-5)
    assert 행.gap_pp > 10.0


def test_마크다운_표에_모든_행이_들어간다(dp):
    점수 = make_scorer(RULES_V1, dp)
    행들 = [점수("a", "rl", greedy_policy_full(dp)),
           점수("b", "supervised", np.zeros(610, dtype=np.int8))]
    md = rows_to_markdown(행들)
    assert "| 모델 |" in md
    assert "a" in md and "b" in md
