"""이 파일은 AI 전략표를 출판 참조표·DP 정답과 3계층으로 대조한다.
입력: AI ChartTable, ReferenceChart, DPResult, 칸별 방문수, 칸별 자연 빈도, 사전등록 표.
출력: AgreementReport(tier_a, tier_b, n_undecided, ev_loss_pp, mismatches).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS, ChartTable
from blackjack_rl.dp.exact import DPResult, greedy_policy_full
from blackjack_rl.reference import TIE_DELTA, ReferenceChart, load_reference_chart
from blackjack_rl.rules import RuleSet
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND

# 왜 0.20인가: DP가 계산한 최선-차선 차가 이만큼 크면 사람이 봐도 뻔한 칸이다.
#   자명한 칸 목록을 손으로 고르면 "유리한 칸만 골랐다"는 반박을 막을 수 없으므로,
#   DP 숫자 하나로 기계적으로 가른다. 이 상수와 아래 trivial_mask 3줄이 슬라이드에 올라간다.
TRIVIAL_DELTA = 0.20

PRE_REGISTERED_PATH = (Path(__file__).resolve().parents[1]
                       / "data" / "expected_mismatch.json")

NOTATION_TO_ACTION: dict[str, int] = {
    "S": STAND, "H": HIT, "D": DOUBLE, "Ds": DOUBLE, "Y": SPLIT,
}

MismatchCause = Literal["deck_model", "rule_diff", "statistical_tie", "undertrained"]

AgreementFn = Callable[[ChartTable, np.ndarray], tuple[float, float]]


def cell_q(dp: DPResult, i: int, j: int) -> np.ndarray:
    """표 (i, j) 칸이 가리키는 상태의 DP 행동가치 4개를 꺼낸다."""
    키 = CHART_ROWS[i].to_key(DEALER_COLS[j], can_double=True)
    return dp.Q[키.total, 키.is_soft, 키.dealer_up,
                키.can_double, 키.can_split, 키.split_depth]


def value_of_notation(q: np.ndarray, notation: str) -> float:
    """표기 하나가 가리키는 행동의 DP 가치를 돌려준다."""
    # 왜: 'N'은 단일 행동이 아니라 '스플릿 말고 나머지 중 최선'이라는 뜻이다.
    #   run_dp.py와 같은 규약을 쓴다.
    if notation == "N":
        return float(np.nanmax(np.delete(q, SPLIT)))
    return float(q[NOTATION_TO_ACTION[notation]])


def best_value(q: np.ndarray) -> float:
    """그 칸에서 DP가 낼 수 있는 최선의 가치."""
    return float(np.nanmax(q))


def chart_gap(dp: DPResult) -> np.ndarray:
    """36x10 칸마다 '최선과 차선의 EV 차'를 계산한다(판정용)."""
    gap = np.zeros((len(CHART_ROWS), len(DEALER_COLS)), dtype=np.float64)
    for i, 행 in enumerate(CHART_ROWS):
        for j in range(len(DEALER_COLS)):
            q = cell_q(dp, i, j)
            if 행.kind == "pair":
                # 왜: 페어 행의 결정은 Y/N 둘 중 하나라 '스플릿 vs 나머지 최선'이 갭이다.
                gap[i, j] = abs(float(q[SPLIT])
                                - float(np.nanmax(np.delete(q, SPLIT))))
            else:
                순위 = np.sort(q[~np.isnan(q)])[::-1]
                gap[i, j] = float(순위[0] - 순위[1])
    return gap


def trivial_mask(dp: DPResult) -> np.ndarray:
    """자명한 칸을 하드코딩하지 않고 DP가 계산한 갭으로 자동 정의한다(실측 186칸)."""
    return chart_gap(dp) >= TRIVIAL_DELTA


def chart_visits(visit_table: np.ndarray) -> np.ndarray:
    """학습이 쌓은 방문 테이블을 36x10 칸별 방문 횟수로 사영한다."""
    # 왜 이 함수가 여기 하나뿐인가: 학습 쪽(npz의 chart_visits)과 일치율 쪽이
    #   같은 사영을 각자 구현하면 dtype이 갈라지고 '방문 0인데 글자가 찍힌 칸'이 생긴다.
    #   train/runner.py가 이 함수를 import해서 쓴다.
    배열 = np.asarray(visit_table)
    if 배열.ndim == 7:
        # 왜: new_visit_table은 행동 축까지 있는 7차원이다. 칸의 방문 횟수는
        #   그 상태에서 내린 모든 결정의 합이므로 행동 축을 더한다.
        배열 = 배열.sum(axis=-1)
    출력 = np.zeros((len(CHART_ROWS), len(DEALER_COLS)), dtype=np.int32)
    for i, 행 in enumerate(CHART_ROWS):
        for j, 딜러 in enumerate(DEALER_COLS):
            k = 행.to_key(딜러, can_double=True)
            출력[i, j] = int(배열[k.total, k.is_soft, k.dealer_up,
                                 k.can_double, k.can_split, k.split_depth])
    return 출력


def chart_margin(Q: np.ndarray) -> np.ndarray:
    """36x10 칸마다 Q(최선) - Q(차선). 히트맵 색으로 쓰는 표시용 확신도다."""
    # 왜 chart_gap과 따로 두는가: 페어 행에서 정의가 다르다. 갭은 'SPLIT vs 나머지',
    #   margin은 '1위 vs 2위'다. 같은 임계값 0.20에서 186칸과 179칸으로 갈리므로,
    #   일치율 분모(판정)는 항상 chart_gap이고 색(표시)만 이 값을 쓴다.
    out = np.zeros((len(CHART_ROWS), len(DEALER_COLS)), dtype=np.float32)
    for i, 행 in enumerate(CHART_ROWS):
        for j, 딜러 in enumerate(DEALER_COLS):
            k = 행.to_key(딜러, can_double=True)
            q = np.asarray(Q[k.total, k.is_soft, k.dealer_up,
                             k.can_double, k.can_split, k.split_depth],
                           dtype=np.float64)
            # 왜 isfinite인가: DP는 불법 행동을 nan으로, 학습 Q는 -inf로 둔다.
            #   한 줄로 두 규약을 같이 거른다. STAND와 HIT은 언제나 합법이라
            #   유한한 값이 항상 2개 이상 있다.
            유한 = np.sort(q[np.isfinite(q)])[::-1]
            out[i, j] = np.float32(유한[0] - 유한[1])
    return out


@dataclass(frozen=True)
class MismatchRow:
    """AI 표기가 참조표와 다른 칸 하나의 기록."""

    cell: str
    dealer: int
    ai: str
    ref: str
    dp_delta: float       # DP 최적 대비 AI 표기의 손해(항상 0 이상)
    visits: int
    natural_freq: float
    ev_loss: float        # natural_freq x dp_delta x 100, 단위 %p
    cause: MismatchCause


@dataclass(frozen=True)
class AgreementReport:
    """발표 헤드라인 네 숫자와 불일치 목록."""

    tier_a: float
    tier_b: float
    n_undecided: int
    ev_loss_pp: float
    mismatches: list[MismatchRow]


def load_pre_registered(path: Path | None = None) -> dict[tuple[str, int], dict]:
    """scripts/run_dp.py --pre-register가 만든 사전등록 JSON을 읽는다."""
    경로 = PRE_REGISTERED_PATH if path is None else path
    본문 = json.loads(경로.read_text(encoding="utf-8"))
    return {(항목["cell"], int(항목["dealer"])): 항목 for 항목 in 본문["entries"]}


def classify(dp_delta: float, entry: dict | None, ai_note: str) -> MismatchCause:
    """불일치 한 칸의 원인을 네 가지 중 하나로 정한다."""
    # 왜 등록 항목을 먼저 보는가: 덱 모형 차이와 규칙 차이는 학습 전에 이미 등록해
    #   두었다. 단, AI가 등록된 DP 표기와 같을 때만 그 사유를 물려받는다.
    #   AI가 DP와도 다르면 그건 새로 생긴 학습 문제다.
    if entry is not None and ai_note == entry["dp"]:
        return entry["cause"]
    if dp_delta < TIE_DELTA:
        return "statistical_tie"
    return "undertrained"


def compare(ai: ChartTable, ref: ReferenceChart, dp: DPResult,
            visits: np.ndarray, freq: np.ndarray,
            pre_registered: dict[tuple[str, int], dict]) -> AgreementReport:
    """AI 표와 참조표를 칸 단위로 대조해 발표용 보고서를 만든다."""
    비자명 = ~trivial_mask(dp)
    미결정 = 0
    a_맞은수 = a_전체수 = b_맞은수 = b_전체수 = 0
    손실합 = 0.0
    불일치들: list[MismatchRow] = []

    for i, 행 in enumerate(CHART_ROWS):
        for j, 딜러 in enumerate(DEALER_COLS):
            ai_표기 = str(ai.notation[i, j])
            if ai_표기 == "":
                # 왜: 한 번도 안 가본 칸은 Q가 초기 노이즈뿐이라 일치율에 넣으면 오염된다.
                미결정 += 1
                continue

            q = cell_q(dp, i, j)
            dp_델타 = best_value(q) - value_of_notation(q, ai_표기)
            손실 = float(freq[i, j]) * dp_델타 * 100.0
            # 왜 모든 칸을 더하는가: AI가 참조표를 그대로 따라 해서 '일치'로 보이지만
            #   DP 기준으로는 손해인 칸도 있다. EV 손실은 참조표가 아니라 DP가 기준이다.
            손실합 += 손실

            ref_표기 = str(ref.notation[i, j])
            등록 = pre_registered.get((행.label, 딜러))
            사전등록됨 = (등록 is not None
                       and ai_표기 == 등록["dp"] and ref_표기 == 등록["ref"])

            if ai_표기 != ref_표기:
                불일치들.append(MismatchRow(
                    cell=행.label, dealer=딜러, ai=ai_표기, ref=ref_표기,
                    dp_delta=dp_델타, visits=int(visits[i, j]),
                    natural_freq=float(freq[i, j]), ev_loss=손실,
                    cause=classify(dp_델타, 등록, ai_표기)))

            if 사전등록됨:
                # 왜: 학습 전에 '원래 다를 칸'으로 등록한 칸은 분모에서도 뺀다.
                #   빼지 않으면 절대 못 맞히는 칸 때문에 일치율 상한이 100% 미만이 된다.
                continue

            a_전체수 += 1
            if 비자명[i, j]:
                b_전체수 += 1
            if ai_표기 == ref_표기:
                a_맞은수 += 1
                if 비자명[i, j]:
                    b_맞은수 += 1

    return AgreementReport(
        tier_a=(a_맞은수 / a_전체수) if a_전체수 else 0.0,
        tier_b=(b_맞은수 / b_전체수) if b_전체수 else 0.0,
        n_undecided=미결정,
        ev_loss_pp=손실합,
        mismatches=불일치들,
    )


def make_agreement_fn(rules: RuleSet, dp: DPResult, *,
                      n_freq: int = 200_000,
                      pre_registered: dict[tuple[str, int], dict] | None = None
                      ) -> AgreementFn:
    """train/runner.run(agreement=...) 자리에 그대로 꽂는 훅을 만든다.

    참조표·사전등록·칸별 자연 빈도는 실행당 한 번만 계산하고 클로저에 가둔다.
    훅이 없으면 산출물의 agree_a/agree_b가 영구히 nan으로 남으므로,
    설계서 §4.1의 '실시간 일치율 카운터'는 이 함수가 꽂혀야 존재한다.
    """
    from blackjack_rl.eval.simulate import EVAL_SEED, make_card_stream, natural_cell_freq

    ref = load_reference_chart()
    등록 = load_pre_registered() if pre_registered is None else pre_registered
    stream = make_card_stream(EVAL_SEED, n_freq, rules)
    freq = natural_cell_freq(greedy_policy_full(dp), rules, stream)

    def 훅(ai: ChartTable, visits: np.ndarray) -> tuple[float, float]:
        보고서 = compare(ai, ref, dp, visits, freq, 등록)
        return float(보고서.tier_a), float(보고서.tier_b)

    return 훅
