"""이 파일은 학습된 정책을 저장하고 되읽을 때 규칙·해시가 검증되는지 확인한다.
입력: 임시 디렉터리와 가짜 정책 배열.
출력: 왕복·해시·규칙 불일치 거부에 대한 pytest 결과.
"""

import json

import numpy as np
import pytest

from blackjack_rl.models.registry import (
    ModelCard,
    ModelRulesMismatch,
    load_policy,
    policy_sha256,
    save_policy,
)
from blackjack_rl.rules import RULES_H17, RULES_V1


def 카드(rules=RULES_V1, name="test_model"):
    return ModelCard(
        name=name, family="supervised", rules_fp=rules.fingerprint(),
        exact_ev=-0.0161, policy_sha256="", n_train_labels=366,
        test_acc=0.877, fit_seconds=0.2, n_params=12345,
        created_at="2026-09-16T00:00:00+00:00",
    )


def test_해시는_64자_16진수이고_내용이_바뀌면_달라진다():
    a = np.zeros(610, dtype=np.int8)
    b = a.copy()
    b[0] = 1
    ha, hb = policy_sha256(a), policy_sha256(b)
    assert len(ha) == 64
    assert all(c in "0123456789abcdef" for c in ha)
    assert ha != hb


def test_같은_정책이면_항상_같은_해시다():
    a = (np.arange(610) % 4).astype(np.int8)
    assert policy_sha256(a) == policy_sha256(a.copy())


def test_저장하고_되읽으면_같다(tmp_path):
    pol = (np.arange(610) % 4).astype(np.int8)
    경로 = save_policy(tmp_path / "m.npz", pol, 카드())
    되읽음, card = load_policy(경로, rules=RULES_V1)
    assert np.array_equal(되읽음, pol)
    assert card.name == "test_model"
    assert card.family == "supervised"


def test_카드가_JSON으로도_남는다(tmp_path):
    pol = np.zeros(610, dtype=np.int8)
    save_policy(tmp_path / "m.npz", pol, 카드())
    실린값 = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert 실린값["rules_fp"] == RULES_V1.fingerprint()
    assert 실린값["policy_sha256"] == policy_sha256(pol)


def test_규칙이_다르면_로딩이_예외다(tmp_path):
    # 왜: 규칙이 다른 모델을 게임 서버가 서빙하면 조용히 틀린 조언을 하게 된다.
    #     경고가 아니라 예외여야 한다.
    save_policy(tmp_path / "m.npz", np.zeros(610, dtype=np.int8), 카드(RULES_V1))
    with pytest.raises(ModelRulesMismatch):
        load_policy(tmp_path / "m.npz", rules=RULES_H17)


def test_정책_길이가_610이_아니면_저장이_예외(tmp_path):
    with pytest.raises(ValueError):
        save_policy(tmp_path / "m.npz", np.zeros(600, dtype=np.int8), 카드())


def test_불법_행동값이_들어있으면_예외(tmp_path):
    나쁜값 = np.zeros(610, dtype=np.int8)
    나쁜값[0] = 7
    with pytest.raises(ValueError):
        save_policy(tmp_path / "m.npz", 나쁜값, 카드())
