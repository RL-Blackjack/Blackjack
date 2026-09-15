"""이 파일은 고정 카드 스트림(CRN)으로 정책의 기대수익을 측정한다.
입력: policy_full(길이 610 행동 배열), RuleSet, make_card_stream이 만든 카드 스트림.
출력: EVResult(ev/se/ci95/n), 필요 표본 크기 정수, 그리고 버그 경보 예외.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from blackjack_rl.cards import CARD_VALUES, ReplayShoe
from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS
from blackjack_rl.dp.exact import (evaluate_policy, evaluate_stochastic,
                                   greedy_policy_full, solve_optimal)
from blackjack_rl.env import BlackjackEnv, Ctx
from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RuleSet
from blackjack_rl.state import (HIT, KEY_INDEX, REACHABLE_KEYS, STAND, StateKey,
                                legal_actions)

# 왜 32장인가: 200만 라운드를 실측했더니 한 라운드가 쓴 카드는 최대 22장이었다
#   (DP 최적 정책 기준. 무작위 정책은 18장). 32장은 그 1.45배라 여유가 충분하고,
#   핸드당 32바이트라 100만 핸드 스트림이 32MB로 메모리에도 들어간다.
CARDS_PER_HAND = 32

# 왜: 평가용 카드 스트림의 기본 시드를 한곳에 못박는다. 설계서 §8.2의 "고정 시드 12345".
EVAL_SEED = 12345

# 왜: scipy가 없으므로 정규분포 분위수를 상수로 적어 둔다.
#   Z_95는 양측 95% 신뢰구간, Z_POWER_80은 검정력 80%에 해당한다.
Z_95 = 1.959964
Z_POWER_80 = 0.841621


class SanityAlarm(Exception):
    """측정 EV가 하우스엣지 바닥보다 좋을 때 올리는 예외."""


BUG_CHECKLIST: tuple[str, ...] = (
    "1) 평가할 때 탐험(엡실론)이 남아 있지 않은가",
    "2) 스플릿 보상을 부모와 자식에서 이중으로 세지 않았는가",
    "3) 더블의 배당 스케일(베팅 2배)이 맞는가",
    "4) 내추럴 블랙잭 3:2 배당을 빠뜨리지 않았는가",
    "5) 평가용 슈와 학습용 슈가 섞이지 않았는가",
)


def make_card_stream(seed: int, n_hands: int, rules: RuleSet) -> np.ndarray:
    """라운드마다 카드 한 뭉치씩을 미리 뽑아 둔 (n_hands, 32) 배열을 만든다."""
    # 왜 2차원인가: 한 줄로 이어 붙이면 두 정책이 카드를 다르게 소비한 순간부터
    #   이후 모든 라운드의 딜이 어긋나 CRN(공통 난수)의 이득이 사라진다.
    #   라운드마다 칸을 잘라 두면 두 정책이 언제나 같은 딜에서 시작한다.
    if rules.deck_mode != "infinite":
        raise ValueError(
            "make_card_stream은 deck_mode='infinite'만 지원한다. "
            "ReplayShoe는 섞지 않으므로 유한덱 슈의 리셔플 시점을 재현할 수 없다.")
    if n_hands <= 0:
        raise ValueError(f"n_hands는 1 이상이어야 한다: {n_hands}")

    rng = np.random.default_rng(seed)
    카드표 = np.asarray(CARD_VALUES, dtype=np.int8)
    # 왜: CARD_VALUES는 13랭크 목록이라 균등하게 뽑으면 10이 저절로 4/13이 된다.
    자리 = rng.integers(0, 카드표.size, size=(n_hands, CARDS_PER_HAND))
    return 카드표[자리]


def greedy_action_fn(policy_full: np.ndarray):
    """policy_full을 환경이 부를 수 있는 ActionFn으로 감싼다(탐험 없음)."""
    배열 = np.asarray(policy_full, dtype=np.int8)
    if 배열.shape != (len(REACHABLE_KEYS),):
        raise ValueError(
            f"policy_full 모양은 ({len(REACHABLE_KEYS)},)여야 한다: {배열.shape}")

    def act(key: StateKey, mask: np.ndarray, ctx: Ctx) -> int:
        번호 = KEY_INDEX.get(key)
        if 번호 is None:
            raise KeyError(f"정책에 없는 상태 키다: {key}")
        행동 = int(배열[번호])
        # 왜: 여기서 막지 않으면 불법 행동이 env까지 흘러가 EV만 이상해진다.
        if not mask[행동]:
            raise ValueError(f"정책이 불법 행동을 골랐다: key={key}, action={행동}")
        return 행동

    return act


def round_nets(policy_full: np.ndarray, rules: RuleSet,
               stream: np.ndarray) -> np.ndarray:
    """스트림의 라운드를 차례로 돌려 라운드별 순손익 배열을 돌려준다."""
    act = greedy_action_fn(policy_full)
    # 왜: BlackjackEnv는 streams를 보관만 하고 play_round에서 쓰지 않는다.
    #   카드는 전부 ReplayShoe에서 나오므로 여기 시드는 결과에 영향이 없다.
    env = BlackjackEnv(rules, ReplayShoe(stream[0]), make_streams(EVAL_SEED))
    결과들 = np.empty(stream.shape[0], dtype=np.float64)
    for i in range(stream.shape[0]):
        env.shoe = ReplayShoe(stream[i])
        결과들[i] = env.play_round(act).net
    return 결과들


@dataclass(frozen=True)
class EVResult:
    """한 정책의 절대 평가 결과."""

    ev: float
    se: float
    ci95: tuple[float, float]
    n: int


def _summarize(표본: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """표본평균·표준오차·95% 신뢰구간을 한 번에 계산한다."""
    n = int(표본.size)
    평균 = float(표본.mean())
    표준오차 = float(표본.std(ddof=1)) / math.sqrt(n)
    return 평균, 표준오차, (평균 - Z_95 * 표준오차, 평균 + Z_95 * 표준오차)


def evaluate(policy_full: np.ndarray, rules: RuleSet,
             stream: np.ndarray) -> EVResult:
    """정책 하나의 라운드당 기대수익을 CRN 스트림으로 측정한다."""
    # 왜 StartSampler 인자가 없는가: 학습은 exploring starts를 쓰지만 평가는 항상
    #   자연 딜이어야 한다(설계서 §8.2). 주석이 아니라 시그니처로 혼입을 막는다.
    순손익 = round_nets(policy_full, rules, stream)
    ev, se, ci95 = _summarize(순손익)
    return EVResult(ev=ev, se=se, ci95=ci95, n=int(순손익.size))


def required_n(half_width: float, sd: float) -> int:
    """반폭 half_width짜리 95% 신뢰구간을 얻는 데 필요한 핸드 수."""
    if half_width <= 0 or sd <= 0:
        raise ValueError("half_width와 sd는 0보다 커야 한다")
    return int(math.ceil((Z_95 * sd / half_width) ** 2))


def required_visits(delta: float, sd: float = 1.1) -> int:
    """행동가치 차 delta를 유의하게 가르는 데 필요한 (s,a)별 방문 횟수."""
    # 왜 이 공식인가: 두 행동의 표본평균 차를 유의수준 5%·검정력 80%로 가르려면
    #   n = 2 (z_0.975 + z_0.80)^2 sd^2 / delta^2 이 필요하다.
    #   sd=1.1을 넣으면 계수가 18.99라서 설계서의 "약 19/delta^2"과 같아진다.
    if delta <= 0 or sd <= 0:
        raise ValueError("delta와 sd는 0보다 커야 한다")
    계수 = 2.0 * (Z_95 + Z_POWER_80) ** 2
    return int(math.ceil(계수 * sd * sd / (delta * delta)))


def sanity_alarm(ev: float, floor: float = -0.003) -> None:
    """측정 EV가 바닥값보다 '좋으면' 예외를 던지고 버그 체크리스트를 보여준다."""
    # 왜 좋으면 막는가: 블랙잭 RL의 버그는 거의 전부 EV를 좋아지게 만든다.
    #   기본전략의 정확 EV가 -0.005108이므로 -0.003보다 좋은 값은 물리적으로 불가능하다.
    if ev > floor:
        줄들 = "\n".join(BUG_CHECKLIST)
        raise SanityAlarm(
            f"측정 EV {ev:+.5f}가 바닥값 {floor:+.5f}보다 좋다. 버그를 의심하라.\n{줄들}")


def baseline_evs(rules: RuleSet) -> list[tuple[str, float]]:
    """비교 기준선 네 개를 학습 결과와 똑같은 DP 정확 평가기로 계산한다."""
    # 왜 여기 한 벌만 두는가: 학습곡선 그림(viz/curves)과 최종 평가표(scripts/evaluate)가
    #   기준선 숫자를 각자 적어 두면 한 저장소 안에서 두 그림이 다른 말을 한다.
    #   실제로 문헌값(-0.1554 / -0.0596)과 이 저장소의 DP 값(-0.160377 / -0.056746)은
    #   다르다 — 문헌값의 규칙 전제(덱 모형·S17/H17)가 우리와 다르기 때문이다.
    n = len(REACHABLE_KEYS)

    최적 = greedy_policy_full(solve_optimal(rules))
    항상스탠드 = np.full(n, STAND, dtype=np.int8)
    # 왜: '딜러 모방'은 플레이어가 딜러 규칙(17 미만이면 히트)을 그대로 따르는 것이다.
    딜러모방 = np.array(
        [HIT if k.total < 17 else STAND for k in REACHABLE_KEYS], dtype=np.int8)

    무작위 = np.zeros((n, 4), dtype=np.float64)
    for i, k in enumerate(REACHABLE_KEYS):
        m = legal_actions(k).astype(np.float64)
        무작위[i] = m / m.sum()

    # 왜 순서를 고정하는가: scripts/evaluate.py의 리포트가 첫 항목을 'DP 상한'으로 읽는다.
    return [
        ("DP 최적(상한)", evaluate_policy(최적, rules)),
        ("딜러 모방", evaluate_policy(딜러모방, rules)),
        ("항상 스탠드", evaluate_policy(항상스탠드, rules)),
        ("무작위(합법 균등)", evaluate_stochastic(무작위, rules)),
    ]


@dataclass(frozen=True)
class PairedResult:
    """같은 카드로 두 정책을 맞붙인 쌍대비교 결과."""

    diff: float
    se: float
    ci95: tuple[float, float]
    divergence_rate: float


def paired_compare(pA: np.ndarray, pB: np.ndarray, rules: RuleSet,
                   stream: np.ndarray) -> PairedResult:
    """두 정책을 같은 카드 스트림으로 돌려 차이(pA - pB)를 잰다."""
    # 왜 쌍대인가: 두 정책이 같은 행동을 한 라운드는 결과가 글자 그대로 같아서
    #   차가 0이 된다. 분산이 '갈라진 라운드'에만 남으므로 독립 스트림 두 벌보다
    #   훨씬 적은 핸드로 같은 정밀도가 나온다(실측 분산 49.5배 절감).
    a = round_nets(pA, rules, stream)
    b = round_nets(pB, rules, stream)
    diff, se, ci95 = _summarize(a - b)
    return PairedResult(diff=diff, se=se, ci95=ci95,
                        divergence_rate=float(np.mean(a != b)))


def _build_cell_of_key() -> dict[StateKey, tuple[int, int]]:
    """36x10 표의 각 칸에 대응하는 StateKey 360개를 미리 만들어 둔다."""
    표 = {}
    for i, 행 in enumerate(CHART_ROWS):
        for j, 딜러 in enumerate(DEALER_COLS):
            # 왜 can_double=True인가: chartspec.project가 칸의 표기를 정할 때
            #   바로 이 키의 Q를 보기 때문이다. 빈도와 표기를 같은 키로 맞춰 둔다.
            표[행.to_key(딜러, can_double=True)] = (i, j)
    return 표


CELL_OF_KEY: dict[StateKey, tuple[int, int]] = _build_cell_of_key()


def natural_cell_freq(policy_full: np.ndarray, rules: RuleSet,
                      stream: np.ndarray) -> np.ndarray:
    """표의 칸마다 '라운드당 평균 몇 번 결정을 내리는가'를 센다."""
    # 왜 이 값이 필요한가: EV 손실 %p는 '틀린 칸의 손해 x 그 칸이 나오는 빈도'다.
    #   하드 16 vs 10은 자주 나오고(0.016535) 8,8 vs 6은 드물게 나오므로, 같은 크기로
    #   틀려도 지갑에 주는 피해가 수십 배 다르다.
    act = greedy_action_fn(policy_full)
    env = BlackjackEnv(rules, ReplayShoe(stream[0]), make_streams(EVAL_SEED))
    센표 = np.zeros((len(CHART_ROWS), len(DEALER_COLS)), dtype=np.float64)
    for i in range(stream.shape[0]):
        env.shoe = ReplayShoe(stream[i])
        결과 = env.play_round(act)
        for 손 in 결과.hands:
            for 키, _행동 in 손.trajectory:
                칸 = CELL_OF_KEY.get(키)
                # 왜 None이 나오는가: 카드를 한 장 더 받은 뒤(can_double=0)의 결정은
                #   표의 칸으로 사영되지 않는다. 그 결정은 세지 않는다.
                #   따라서 ev_loss_pp는 엄밀히는 '첫 결정 기준 하한'이다.
                if 칸 is not None:
                    센표[칸] += 1.0
    return 센표 / float(stream.shape[0])
