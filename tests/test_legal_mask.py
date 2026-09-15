"""이 파일은 행동 마스크 테이블 LEGAL이 정확한지 검증한다.
입력: blackjack_rl.state 모듈의 LEGAL / _build_legal_table / legal_actions
출력: pytest 통과 또는 실패 메시지
"""

from dataclasses import replace

import numpy as np
import pytest

from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import (
    ACTION_NAMES,
    ACTION_NAMES_KO,
    ACTIONS,
    DOUBLE,
    HIT,
    LEGAL,
    Q_SHAPE,
    SPLIT,
    STAND,
    StateKey,
    _build_legal_table,
    legal_actions,
)


def test_action_constants():
    # 행동 인덱스는 영원히 고정이다. 바뀌면 저장된 npz가 전부 무의미해진다.
    assert (STAND, HIT, DOUBLE, SPLIT) == (0, 1, 2, 3)
    assert ACTIONS == (0, 1, 2, 3)
    assert len(ACTION_NAMES) == 4
    assert len(ACTION_NAMES_KO) == 4


def test_state_key_default_split_depth():
    key = StateKey(16, 0, 10, 0, 0)
    assert key.split_depth == 0
    assert tuple(key) == (16, 0, 10, 0, 0, 0)


def test_legal_shape():
    assert LEGAL.shape == Q_SHAPE
    assert LEGAL.dtype == np.bool_


def test_legal_matches_pure_function():
    # (1) 사전계산 테이블이 순수 함수의 결과와 전 키에서 완전히 같아야 한다.
    assert np.array_equal(LEGAL, _build_legal_table(RULES_V1))


def test_legal_table_does_not_depend_on_other_rules():
    # 마스크는 키만의 함수다. 딜러 규칙이나 DAS를 바꿔도 테이블은 같아야 한다.
    h17 = replace(RULES_V1, dealer_hits_soft_17=True)
    no_das = replace(RULES_V1, double_after_split=False)
    assert np.array_equal(LEGAL, _build_legal_table(h17))
    assert np.array_equal(LEGAL, _build_legal_table(no_das))


def test_stand_and_hit_always_legal():
    # (2) STAND와 HIT은 모든 키에서 항상 합법이다.
    #     덕분에 "합법 행동 0개" 예외가 구조적으로 생길 수 없다.
    assert LEGAL[..., STAND].all()
    assert LEGAL[..., HIT].all()


def test_double_legal_exactly_when_can_double():
    # (3) DOUBLE은 can_double 축이 1일 때만 합법이다.
    assert not LEGAL[:, :, :, 0, :, :, DOUBLE].any()
    assert LEGAL[:, :, :, 1, :, :, DOUBLE].all()


def test_split_legal_exactly_when_can_split():
    # (3) SPLIT은 can_split 축이 1일 때만 합법이다.
    assert not LEGAL[:, :, :, :, 0, :, SPLIT].any()
    assert LEGAL[:, :, :, :, 1, :, SPLIT].all()


def test_legal_actions_returns_view_not_copy():
    # (4) 이것이 3~5배 성능의 근거다. 배열을 새로 만들지 않고 뷰만 돌려준다.
    key = StateKey(16, 0, 10, 1, 1, 0)
    mask = legal_actions(key)
    assert np.shares_memory(mask, LEGAL)
    assert mask.shape == (4,)
    assert mask.dtype == np.bool_


def test_legal_actions_view_for_many_keys():
    # 특정 키에서만 우연히 뷰인 것이 아니라 전 범위에서 뷰여야 한다.
    for total in range(4, 22):
        for dealer_up in range(2, 12):
            key = StateKey(total, 0, dealer_up, 1, 0, 0)
            assert np.shares_memory(legal_actions(key), LEGAL)


def test_legal_actions_values():
    assert legal_actions(StateKey(16, 0, 10, 0, 0, 0)).tolist() == [True, True, False, False]
    assert legal_actions(StateKey(16, 0, 10, 1, 0, 0)).tolist() == [True, True, True, False]
    assert legal_actions(StateKey(16, 0, 10, 1, 1, 0)).tolist() == [True, True, True, True]


def test_legal_table_is_readonly():
    # 뷰를 돌려주므로, 호출자가 실수로 한 칸을 고치면 전역 마스크가 영구 오염된다.
    # 그래서 LEGAL은 읽기 전용으로 잠겨 있어야 한다.
    with pytest.raises(ValueError):
        legal_actions(StateKey(16, 0, 10, 1, 1, 0))[SPLIT] = False
