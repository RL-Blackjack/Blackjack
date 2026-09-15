"""이 파일은 정확 DP와 시뮬레이터가 같은 게임인지 대조한다.
입력: conftest의 rules·dp 픽스처
출력: 세 가지 검증(전체 EV / 개별 셀 / DP 근사 오차)
"""

import math
from dataclasses import replace

import pytest

from blackjack_rl.cards import make_shoe
from blackjack_rl.dp.exact import action_values, greedy_policy_full
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.hand import Hand
from blackjack_rl.rng import make_streams
from blackjack_rl.starts import StartState
from blackjack_rl.state import KEY_INDEX

SEED = 20260915


def make_dp_policy(dp):
    """DP 최적 정책을 환경이 부를 수 있는 ActionFn으로 감싼다."""
    policy = greedy_policy_full(dp)

    def act(key, mask, ctx):
        action = int(policy[KEY_INDEX[key]])
        # 왜 여기서 검사하는가: DP 정책이 불법 행동을 고르면 상태 키 설계나 마스크
        #   둘 중 하나가 깨졌다는 뜻인데, 조용히 넘어가면 EV만 이상하게 나온다.
        assert mask[action], f"DP 정책이 불법 행동을 골랐다: key={key}, action={action}"
        return action

    return act


def run_rounds(rules, act, n, seed, start=None, only_decided=False):
    """n판을 돌려 (평균, 표준오차)를 돌려준다.

    only_decided=True면 플레이어가 한 번도 결정하지 못한 판을 평균에서 뺀다.
    왜 필요한가: 딜러가 블랙잭이면 피크 단계에서 라운드가 즉시 끝나 플레이어는
    행동할 기회조차 없다. 그런데 dp.action_values(key)는 '이 상태에서 이 행동을
    했을 때의 가치'이므로 정의상 '딜러가 블랙잭이 아닌' 조건부 값이다. 두 분포를
    섞어서 비교하면 딜러 업카드가 A나 10인 칸에서만 어긋난다(P(딜러 BJ)가 그때만
    0이 아니므로). 실제로 하드 16 vs 10에서 무조건부 -0.5758 vs 조건부 -0.5404로
    0.035 차이가 났고, 이는 (1/13)x(-1) + (12/13)x(-0.5404)와 정확히 일치했다.
    """
    streams = make_streams(seed)
    env = BlackjackEnv(rules, make_shoe(rules, streams.deal), streams)
    total = 0.0
    total_sq = 0.0
    세어본판 = 0
    for _ in range(n):
        r = env.play_round(act, start=start)
        if only_decided and r.n_decisions == 0:
            continue
        net = r.net
        total += net
        total_sq += net * net
        세어본판 += 1
    mean = total / 세어본판
    var = max(total_sq / 세어본판 - mean * mean, 0.0)
    return mean, math.sqrt(var / 세어본판)


@pytest.mark.slow
def test_DP최적정책의_시뮬EV가_DP정확값과_3시그마_안에서_일치한다(rules, dp):
    n = 2_000_000
    mean, se = run_rounds(rules, make_dp_policy(dp), n, SEED)
    print(f"\n시뮬 EV = {mean:+.5f} ± {se:.5f} (n={n:,})")
    print(f"DP  EV = {dp.ev_initial:+.5f}")
    print(f"차이   = {abs(mean - dp.ev_initial) / se:.2f} sigma")
    assert abs(mean - dp.ev_initial) < 3 * se


# (플레이어 카드, 딜러 업카드, 사람이 읽는 이름)
CELLS = [
    ([10, 6], 10, "하드 16 vs 10"),
    ([10, 2], 3, "하드 12 vs 3"),
    ([1, 7], 9, "소프트 A,7 vs 9"),
]


@pytest.mark.slow
@pytest.mark.parametrize("cards,up,name", CELLS)
def test_개별_셀의_행동가치가_시뮬_추정치와_3시그마_안에서_일치한다(rules, dp, cards, up, name):
    key = Hand(cards=list(cards)).state_key(up, rules, hands_in_round=1)
    dp_values = action_values(key, rules)
    policy = greedy_policy_full(dp)
    n = 200_000

    for action in range(4):
        if math.isnan(dp_values[action]):
            continue  # 불법 행동은 DP가 nan을 준다

        def act(k, mask, ctx, _first=action):
            # 첫 결정만 강제하고 그 뒤는 DP 최적 플레이로 이어간다.
            return _first if k == key else int(policy[KEY_INDEX[k]])

        start = StartState(player_cards=list(cards), dealer_up=up, forced_first_action=action)
        # only_decided=True: 딜러 블랙잭으로 플레이어가 행동하지 못한 판은 빼야
        # DP의 조건부 행동가치와 같은 분포가 된다(run_rounds의 설명 참조).
        mean, se = run_rounds(rules, act, n, SEED + action, start=start, only_decided=True)
        print(f"\n{name} 행동{action}: 시뮬 {mean:+.4f}±{se:.4f} vs DP {dp_values[action]:+.4f}")
        assert abs(mean - dp_values[action]) < 3 * se


@pytest.mark.slow
def test_라운드_손수_상한을_DP가_근사한_오차를_수치로_남긴다(rules, dp):
    """DP는 손별 깊이만 보므로 '라운드 최대 4손'을 근사한다. 그 크기를 직접 잰다."""
    n = 1_000_000
    act = make_dp_policy(dp)
    tight, se_t = run_rounds(rules, act, n, SEED)
    loose, se_l = run_rounds(replace(rules, max_hands_per_round=8), act, n, SEED)

    gap = abs(tight - loose)
    se = math.sqrt(se_t ** 2 + se_l ** 2)
    print(f"\n손 수 상한 4손 EV = {tight:+.5f}")
    print(f"손 수 상한 8손 EV = {loose:+.5f}")
    print(f"DP 근사 오차      = {gap:.5f} ± {se:.5f}  ({gap * 100:.3f}%p)")

    # 왜 0.001인가: 라운드당 4손을 넘기려면 페어를 연달아 3번 이상 받아야 하는데
    #   그 확률이 매우 낮다. 이 값이 크게 나오면 스플릿 상한 구현이 의심스럽다.
    assert gap < 0.001
