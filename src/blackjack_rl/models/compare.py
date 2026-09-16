"""이 파일은 어떤 모델이든 같은 자로 재서 비교표 한 줄로 만든다.
입력: 정책 배열(int8[610])과 모델 메타데이터.
출력: 정확 EV·일치율·비용을 담은 ModelRow와 마크다운 표.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackjack_rl.chartspec import ChartTable, project
from blackjack_rl.dp.exact import DPResult, evaluate_policy
from blackjack_rl.eval.agreement import compare as 일치율비교
from blackjack_rl.eval.agreement import load_pre_registered, load_reference_chart
from blackjack_rl.eval.simulate import EVAL_SEED, make_card_stream, natural_cell_freq
from blackjack_rl.rules import RuleSet
from blackjack_rl.state import LEGAL, Q_SHAPE, REACHABLE_KEYS

FREQ_HANDS: int = 200_000


def policy_to_q(policy_full: np.ndarray) -> np.ndarray:
    """정책을 Q 배열로 되돌린다. 고른 행동 1.0 / 나머지 합법 0.0 / 불법 nan.

    # 왜 이런 변환이 필요한가: chartspec.project()는 Q를 받도록 만들어졌는데
    #   지도학습 모델은 정책만 내놓는다. 이렇게 채우면 project()가 그대로
    #   동작하고 D/Ds 표기까지 올바르게 나온다(can_double을 두 번 조회하므로).
    """
    Q = np.full(Q_SHAPE, np.nan)
    for i, key in enumerate(REACHABLE_KEYS):
        자리 = (key.total, key.is_soft, key.dealer_up,
               key.can_double, key.can_split, key.split_depth)
        Q[자리] = np.where(LEGAL[자리], 0.0, np.nan)
        Q[자리 + (int(policy_full[i]),)] = 1.0
    return Q


def policy_to_chart(policy_full: np.ndarray, rules: RuleSet) -> ChartTable:
    """정책을 36x10 전략표로 사영한다."""
    return project(policy_to_q(policy_full), rules)


@dataclass(frozen=True)
class ModelRow:
    """비교표 한 줄. 모델 종류와 무관하게 같은 항목을 담는다."""

    name: str
    family: str
    exact_ev: float
    gap_pp: float
    tier_a: float
    tier_b: float
    n_undecided: int
    n_train_labels: int
    test_acc: float
    fit_seconds: float
    n_params: int


def make_scorer(rules: RuleSet, dp: DPResult):
    """참조표와 칸별 빈도를 한 번만 계산해 두고 채점 함수를 돌려준다."""
    # 왜 클로저인가: 참조표 로딩과 20만 핸드 빈도 계산은 모델마다 다시 할 이유가
    #   없다. 60개 모델을 잴 때 이 한 줄이 시간을 수십 배 줄인다.
    참조표 = load_reference_chart()
    사전등록 = load_pre_registered()
    빈도 = natural_cell_freq(
        np.array([int(np.nanargmax(dp.Q[k])) for k in REACHABLE_KEYS], dtype=np.int8),
        rules, make_card_stream(EVAL_SEED, FREQ_HANDS, rules))

    def score(name: str, family: str, policy_full: np.ndarray, *,
              n_train_labels: int = 0, test_acc: float = float("nan"),
              fit_seconds: float = 0.0, n_params: int = 0) -> ModelRow:
        정책 = np.ascontiguousarray(policy_full, dtype=np.int8)
        ev = float(evaluate_policy(정책, rules))
        표 = policy_to_chart(정책, rules)
        # 왜 visits를 전부 1로 두는가: 지도학습 모델에는 방문 횟수 개념이 없다.
        #   미결정은 REACHABLE_KEYS에 없는 칸에서만 생겨야 한다.
        보고서 = 일치율비교(표, 참조표, dp, np.ones((36, 10), dtype=np.int32),
                        빈도, 사전등록)
        return ModelRow(
            name=name, family=family, exact_ev=ev,
            gap_pp=(dp.ev_initial - ev) * 100.0,
            tier_a=float(보고서.tier_a), tier_b=float(보고서.tier_b),
            n_undecided=int(보고서.n_undecided),
            n_train_labels=n_train_labels, test_acc=test_acc,
            fit_seconds=fit_seconds, n_params=n_params,
        )

    return score


def rows_to_markdown(rows: list[ModelRow]) -> str:
    """비교표를 마크다운으로 만든다. 보고서와 대시보드가 같은 표를 쓴다."""
    줄 = [
        "| 모델 | 계열 | 정확 EV | DP와의 격차 | 결정 가능 칸 일치 | 비자명 일치 | 미결정 | 훈련 레이블 | 시험 정확도 | 학습 시간 | 파라미터 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        정확도 = "—" if np.isnan(r.test_acc) else f"{r.test_acc:.3f}"
        줄.append(
            f"| {r.name} | {r.family} | {r.exact_ev * 100:+.4f}% | {r.gap_pp:.4f}%p | "
            f"{r.tier_a * 100:.2f}% | {r.tier_b * 100:.2f}% | {r.n_undecided} | "
            f"{r.n_train_labels} | {정확도} | {r.fit_seconds:.2f}초 | {r.n_params:,} |")
    return "\n".join(줄)
