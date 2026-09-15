"""이 파일은 도달 가능 상태 목록과 Q·방문 테이블 초기화를 검증한다.
입력: blackjack_rl.state 모듈의 REACHABLE_KEYS / KEY_INDEX / new_q_table / new_visit_table
출력: pytest 통과 또는 실패 메시지
"""

import numpy as np
import pytest

from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import (
    KEY_INDEX,
    LEGAL,
    Q_SHAPE,
    REACHABLE_KEYS,
    STAND,
    StateKey,
    _build_legal_table,
    key_flat_index,
    new_q_table,
    new_visit_table,
)


def test_reachable_key_count():
    # (1) 손으로 센 값과 정확히 같아야 한다.
    #     딜러 업카드 10종 × 손 상태 61종 = 610.
    #     (설계서의 "약 1,100개"는 어림값이었고, 실제로 세면 610개다.)
    assert len(REACHABLE_KEYS) == 610


def test_reachable_group_counts():
    pairs = [k for k in REACHABLE_KEYS if k.can_split == 1]
    first_two = [k for k in REACHABLE_KEYS if k.can_double == 1 and k.can_split == 0]
    after_hit = [k for k in REACHABLE_KEYS if k.can_double == 0]
    assert len(pairs) == 100       # 페어 10종 × 10
    assert len(first_two) == 260   # 첫 두 장이지만 페어 아님 26종 × 10
    assert len(after_hit) == 250   # 한 번이라도 히트한 뒤 25종 × 10


def test_no_split_without_double():
    # 페어는 언제나 '첫 두 장'이므로 can_split=1이면 can_double=1이어야 한다.
    for key in REACHABLE_KEYS:
        if key.can_split == 1:
            assert key.can_double == 1


def test_key_ranges():
    for key in REACHABLE_KEYS:
        assert 4 <= key.total <= 21
        assert key.is_soft in (0, 1)
        assert 2 <= key.dealer_up <= 11
        assert key.split_depth == 0   # 기본 모드는 split_depth를 키에 넣지 않는다


def test_specific_keys():
    assert StateKey(16, 0, 10, 0, 0, 0) in KEY_INDEX      # 히트 후 하드 16 vs 10
    assert StateKey(16, 0, 10, 1, 1, 0) in KEY_INDEX      # 8,8 vs 10
    assert StateKey(12, 1, 6, 1, 1, 0) in KEY_INDEX       # A,A vs 6
    # A,A는 언제나 스플릿할 수 있으므로 '스플릿 못 하는 소프트 12'는 존재하지 않는다.
    assert StateKey(12, 1, 6, 1, 0, 0) not in KEY_INDEX
    # 두 장으로 하드 21을 만들 수는 없다(A+10은 소프트 21이다).
    assert StateKey(21, 0, 10, 1, 0, 0) not in KEY_INDEX


def test_keys_sorted_and_unique():
    assert len(set(REACHABLE_KEYS)) == len(REACHABLE_KEYS)
    assert list(REACHABLE_KEYS) == sorted(REACHABLE_KEYS)


def test_key_flat_index():
    for i, key in enumerate(REACHABLE_KEYS):
        assert key_flat_index(key) == i
    with pytest.raises(KeyError):
        key_flat_index(StateKey(12, 1, 6, 1, 0, 0))


def test_illegal_is_neg_inf():
    # (2) 불법 (s,a)는 -inf로 초기화된다.
    q = new_q_table(RULES_V1, np.random.default_rng(0))
    legal = _build_legal_table(RULES_V1)
    assert q.shape == Q_SHAPE
    assert np.all(np.isneginf(q[~legal]))
    assert np.all(np.isfinite(q[legal]))


def test_noise_is_tiny():
    q = new_q_table(RULES_V1, np.random.default_rng(0))
    legal = _build_legal_table(RULES_V1)
    assert np.max(np.abs(q[legal])) <= 1e-6


def test_argmax_not_always_stand():
    # (3) 이것이 히트맵 글자 깜빡임과 일치율 왜곡을 동시에 막는 장치다.
    #     전부 0이면 argmax가 항상 인덱스 0(STAND)으로 고정된다.
    q = new_q_table(RULES_V1, np.random.default_rng(0))
    chosen = [int(np.argmax(q[key])) for key in REACHABLE_KEYS]
    assert len(set(chosen)) >= 3
    assert chosen.count(STAND) < len(chosen) * 0.6
    zeros = np.zeros(Q_SHAPE)
    assert all(int(np.argmax(zeros[key])) == STAND for key in REACHABLE_KEYS)


def test_argmax_is_legal():
    # -inf 덕분에 미학습 상태에서도 불법 행동이 선택될 수 없다.
    q = new_q_table(RULES_V1, np.random.default_rng(1))
    for key in REACHABLE_KEYS:
        assert LEGAL[key][int(np.argmax(q[key]))]


def test_same_seed_same_table():
    # (4) 같은 시드면 같은 Q 테이블. 재현성 5단 장치의 맨 아래 칸이다.
    a = new_q_table(RULES_V1, np.random.default_rng(42))
    b = new_q_table(RULES_V1, np.random.default_rng(42))
    c = new_q_table(RULES_V1, np.random.default_rng(43))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_visit_table():
    n = new_visit_table(RULES_V1)
    assert n.shape == Q_SHAPE
    assert n.dtype == np.int32
    assert n.sum() == 0
