"""이 파일은 블랙잭의 상태 키와 행동, 그리고 행동 마스크 테이블을 정의한다.
입력: RuleSet(규칙)
출력: StateKey / 행동 상수 / 사전계산된 LEGAL 마스크 테이블
"""

from collections import deque
from typing import NamedTuple

import numpy as np

from blackjack_rl.rules import RULES_V1, RuleSet

# ─────────────────────────────────────────────────────────────
# ★ 이 파일은 학생이 직접 타이핑하는 파일 ①이다. 복사·붙여넣기 금지.
#
#   지배 원칙: 행동 마스크가 의존하는 모든 정보는 상태 키 안에 있다.
#
# 발표 질문 "상태를 왜 그렇게 정했나?"에 대한 답:
#   1) can_double(첫 두 장인가)과 can_split(같은 값 두 장 + 스플릿 여유)을
#      키에 넣었기 때문에 legal_actions(key)가 '키만의 순수 함수'가 된다.
#      키 밖의 정보(손에 든 카드 장수, 라운드의 다른 손)를 보지 않아도
#      합법 행동이 정해지므로, 매번 계산하지 않고 미리 만들어 둘 수 있다.
#   2) 페어의 랭크는 별도 차원으로 넣지 않는다. 같은 랭크 두 장의 합은
#      하드 2r이고 유일한 소프트 페어는 (A,A)=소프트 12이므로
#      (total, is_soft, can_split)에서 랭크가 유일하게 복원된다.
#      예: 하드 16 + can_split=1 이면 반드시 8,8이다. 차원을 늘리면 낭비다.
#   3) 버스트(22 이상)는 상태가 아니라 종단이므로 total은 4..21만 쓴다.
# ─────────────────────────────────────────────────────────────

STAND = 0
HIT = 1
DOUBLE = 2
SPLIT = 3
ACTIONS = (STAND, HIT, DOUBLE, SPLIT)
ACTION_NAMES = ("S", "H", "D", "P")
ACTION_NAMES_KO = ("스탠드", "히트", "더블", "스플릿")


class StateKey(NamedTuple):
    total: int        # 4..21 (22 이상은 버스트=종단이라 상태로 존재하지 않는다)
    is_soft: int      # 0/1 (에이스를 11로 세고 있으면 1)
    dealer_up: int    # 2..11 (11 = 에이스)
    can_double: int   # 0/1 (첫 두 장인가)
    can_split: int    # 0/1 (같은 값 두 장 + 스플릿 여유가 있는가)
    split_depth: int = 0   # include_split_depth_in_state=False면 항상 0


# 축 순서: total, is_soft, dealer_up, can_double, can_split, split_depth, action
# 왜: split_depth 축을 기본 모드에서도 길이 4로 고정해 둔다. 어블레이션 때
#     배열 모양이 바뀌면 저장 포맷과 인덱싱 코드가 전부 분기되기 때문이다.
Q_SHAPE = (22, 2, 12, 2, 2, 4, 4)


def _build_legal_table(rules: RuleSet) -> np.ndarray:
    """모든 키에 대해 합법 행동을 계산한 bool 테이블을 새로 만들어 돌려준다."""
    table = np.zeros(Q_SHAPE, dtype=bool)
    for total in range(22):
        for is_soft in range(2):
            for dealer_up in range(12):
                for can_double in range(2):
                    for can_split in range(2):
                        for split_depth in range(4):
                            cell = (total, is_soft, dealer_up,
                                    can_double, can_split, split_depth)
                            # STAND와 HIT은 언제나 합법이다.
                            table[cell][STAND] = True
                            table[cell][HIT] = True
                            # DOUBLE/SPLIT의 가능 여부는 이미 키 안에 들어 있다.
                            table[cell][DOUBLE] = bool(can_double)
                            table[cell][SPLIT] = bool(can_split)
    return table


# 왜: 부팅 시 1회만 계산한다. 마스크는 키만의 순수 함수라 규칙이 달라져도
#     테이블 내용이 같으므로(규칙 의존성은 can_double/can_split 계산 쪽이 이미 흡수)
#     전역 테이블 하나로 충분하다.
LEGAL = _build_legal_table(RULES_V1)
# 왜: legal_actions가 뷰를 돌려주므로, 호출자가 실수로 한 칸을 고치면
#     전역 마스크가 영구히 오염된다. 아예 쓰기를 막아 둔다.
LEGAL.flags.writeable = False


def legal_actions(key: StateKey) -> np.ndarray:
    """키 하나에 대한 길이 4 합법 마스크를 돌려준다."""
    # 왜: np.array(...)로 매번 만들면 결정 1회당 배열 할당이 1~2μs 붙는다.
    #     에피소드당 결정이 1.3~1.8회 × 수천만 에피소드면 이게 전체 병목이다.
    #     여기서는 미리 만든 테이블의 뷰를 돌려주므로 할당이 0이다.
    return LEGAL[key]


# ─────────────────────────────────────────────────────────────
# 발표 질문 "그럼 상태가 정말 Markov인가?"에 대한 답:
#   마스크는 순수 함수(참)이지만, 상태는 근사적으로만 Markov(정직한 인정)이다.
#   기본 모드는 split_depth를 키에서 빼기 때문에 "앞으로 2번 더 쪼갤 수 있다"와
#   "1번 더"가 같은 칸에 별칭(aliasing)된다. 그래서 include_split_depth_in_state는
#   '여유 시 옵션'이 아니라 필수 대조 실험이고, 플래그를 켜면 아래 BFS가
#   같은 코드로 키를 2,230개까지 늘려 준다. 분기 코드는 한 줄도 늘지 않는다.
#
# 도달 가능 키가 610개인 이유 (딜러 업카드 10종 × 손 상태 61종):
#   - 페어 (can_double=1, can_split=1) 10종
#       하드 4,6,8,10,12,14,16,18,20 과 소프트 12(A,A)
#   - 첫 두 장이지만 페어가 아님 (can_double=1, can_split=0) 26종
#       하드 4..20(17종) + 소프트 13..21(9종)
#       하드 4와 20은 2,2 / 10,10 이 스플릿 한도에 걸려 더 못 쪼개는 경우다.
#       소프트 12(A,A)는 여기 없다 — A,A는 언제나 쪼갤 수 있기 때문이다.
#   - 한 번이라도 히트한 뒤 (can_double=0, can_split=0) 25종
#       하드 6..21(16종) + 소프트 13..21(9종)
#   설계서의 "약 1,100개"는 어림값이었고, 실제로 세어 보면 610개다.
# ─────────────────────────────────────────────────────────────
# 왜: 랭크 집합만 필요하고 뽑기 확률은 필요 없으므로 cards.py를 import하지 않는다.
#     상태 모듈을 의존성 없이 두면 import 순환이 원천적으로 생길 수 없다.
_RANKS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def _hit(total: int, is_soft: int, card: int) -> tuple[int, int]:
    """카드 한 장을 받은 뒤의 (합계, 소프트여부)를 돌려준다."""
    if card == 1:
        # 에이스는 11로 셀 수 있으면 11로 센다.
        if total + 11 <= 21:
            return total + 11, 1
        new_total = total + 1
        if new_total > 21 and is_soft == 1:
            return new_total - 10, 0
        return new_total, is_soft
    new_total = total + card
    # 왜: 11로 세던 에이스를 1로 내리는 것은 딱 한 번만 가능하다.
    #     두 번째 에이스는 애초에 1로 세어 넣었기 때문이다.
    if new_total > 21 and is_soft == 1:
        return new_total - 10, 0
    return new_total, is_soft


def _two_cards(card_a: int, card_b: int) -> tuple[int, int]:
    """빈 손에서 두 장을 받은 뒤의 (합계, 소프트여부)를 돌려준다."""
    total, is_soft = _hit(0, 0, card_a)
    return _hit(total, is_soft, card_b)


def _can_split_flag(rules: RuleSet, rank: int, depth: int) -> int:
    """랭크 rank인 페어를 깊이 depth에서 더 쪼갤 수 있으면 1."""
    if depth >= rules.max_split_depth:
        return 0
    if rank == 1 and depth >= 1 and not rules.resplit_aces:
        return 0
    return 1


def _enumerate_reachable(rules: RuleSet) -> tuple[StateKey, ...]:
    """실제 게임에서 도달 가능한 상태 키를 BFS로 전수 열거한다."""
    # 왜: 라운드 전체 손 수 상한(max_hands_per_round=4)은 손별 정보가 아니라
    #     라운드 공유 자원이라 키에 넣을 수 없다. 기본 규칙에서는 손별 깊이 3과
    #     라운드 4손이 동시에 걸리므로 깊이만 따져도 키 집합이 같다.
    seen = set()
    queue = deque()
    # 출발점: 처음 두 장을 받은 모든 손
    for card_a in _RANKS:
        for card_b in _RANKS:
            total, is_soft = _two_cards(card_a, card_b)
            if card_a == card_b:
                can_split = _can_split_flag(rules, card_a, 0)
            else:
                can_split = 0
            queue.append((total, is_soft, 1, can_split, 0))
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        total, is_soft, can_double, can_split, depth = node
        # HIT: 한 장 더 받으면 더블도 스플릿도 불가능해진다. 버스트는 종단이라 버린다.
        for card in _RANKS:
            next_total, next_soft = _hit(total, is_soft, card)
            if next_total <= 21:
                queue.append((next_total, next_soft, 0, 0, depth))
        # SPLIT: 자식 손은 '랭크 한 장 + 새 카드 한 장'짜리 두 장 손이 된다.
        if can_split == 1:
            # 왜: 하드 2r의 랭크는 total//2, 유일한 소프트 페어는 (A,A)이므로
            #     키만으로 페어의 랭크가 유일하게 복원된다.
            rank = 1 if is_soft == 1 else total // 2
            # 왜: 에이스를 스플릿하면 자식은 카드 한 장만 받고 즉시 끝나므로
            #     자식에게는 결정 상태가 아예 없다.
            ace_one_card = (rank == 1 and rules.split_aces_one_card)
            if not ace_one_card:
                child_depth = depth + 1
                for card in _RANKS:
                    child_total, child_soft = _two_cards(rank, card)
                    child_double = 1 if rules.double_after_split else 0
                    if card == rank:
                        child_split = _can_split_flag(rules, rank, child_depth)
                    else:
                        child_split = 0
                    queue.append((child_total, child_soft,
                                  child_double, child_split, child_depth))
    keys = set()
    for total, is_soft, can_double, can_split, depth in seen:
        if rules.include_split_depth_in_state:
            key_depth = depth
        else:
            key_depth = 0
        for dealer_up in range(2, 12):
            keys.add(StateKey(total, is_soft, dealer_up,
                              can_double, can_split, key_depth))
    # 왜: 정렬해서 고정한다. 이 순서가 policy_full 배열의 열 순서이므로
    #     실행할 때마다 달라지면 저장된 스냅샷을 다시 읽을 수 없다.
    return tuple(sorted(keys))


REACHABLE_KEYS = _enumerate_reachable(RULES_V1)
KEY_INDEX = {key: i for i, key in enumerate(REACHABLE_KEYS)}


def key_flat_index(key: StateKey) -> int:
    """상태 키를 REACHABLE_KEYS 안의 0부터 시작하는 번호로 바꾼다."""
    if key not in KEY_INDEX:
        raise KeyError(f"도달 불가능한 상태 키입니다: {key}")
    return KEY_INDEX[key]


def new_q_table(rules: RuleSet, rng) -> np.ndarray:
    """합법 (s,a)는 아주 작은 난수로, 불법 (s,a)는 -inf로 채운 Q 테이블."""
    legal = _build_legal_table(rules)
    # 왜: 마스크와 상관없이 항상 같은 개수의 난수를 뽑는다. 그래야 규칙을 바꿔도
    #     시드가 같으면 난수 소비 순서가 흔들리지 않는다.
    noise = rng.uniform(-1e-6, 1e-6, size=Q_SHAPE)
    # 왜: 불법 행동은 -inf라서 argmax가 절대 고르지 않고, 절대 갱신하지도 않는다.
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float64)
    q[legal] = noise[legal]
    return q


def new_visit_table(rules: RuleSet) -> np.ndarray:
    """Q와 같은 모양의 방문 횟수 테이블(전부 0)."""
    return np.zeros(Q_SHAPE, dtype=np.int32)
