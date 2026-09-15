"""이 파일은 스플릿 크레딧 할당(설계서 §3.4)이 정확한지 검사한다.
입력: 손으로 조립한 RoundResult.
출력: pytest 통과/실패."""

from blackjack_rl.env import HandRecord, RoundResult
from blackjack_rl.returns import Transition, assert_invariants, compute_transitions
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND, StateKey


def test_설계서_손계산_예제_8_8_대_딜러6():
    """8,8 vs 딜러 6에서 스플릿. 첫 손 8→13→스탠드 +1, 둘째 손 8→18에서 더블 -2.
    그러면 split 노드의 리턴은 -1, 13 vs 6·STAND는 +1, 18 vs 6·DOUBLE은 -2다."""
    k_split = StateKey(16, 0, 6, 1, 1, 0)
    k13 = StateKey(13, 0, 6, 1, 0, 0)
    k18 = StateKey(18, 0, 6, 1, 0, 0)

    분기 = HandRecord(
        trajectory=[(k_split, SPLIT)],
        parent=None, parent_step=None, bet=1.0, result=0.0, subtree_result=-1.0,
    )
    첫손 = HandRecord(
        trajectory=[(k13, STAND)],
        parent=0, parent_step=0, bet=1.0, result=1.0, subtree_result=1.0,
    )
    둘째손 = HandRecord(
        trajectory=[(k18, DOUBLE)],
        parent=0, parent_step=0, bet=2.0, result=-2.0, subtree_result=-2.0,
    )
    # 딜러 카드는 이 검사와 무관하다(설계서 §3.4의 숫자를 그대로 옮긴 예제다).
    r = RoundResult(
        hands=[분기, 첫손, 둘째손], dealer_up=6, dealer_cards=[6, 10, 5],
        net=-1.0, n_decisions=3, tc_at_deal=0.0,
    )

    assert_invariants(r)
    trs = compute_transitions(r)
    assert len(trs) == 3

    by_key = {}
    for tr in trs:
        by_key[tr.key] = tr

    assert by_key[k_split].action == SPLIT
    assert by_key[k_split].mc_return == -1.0          # 자손 합 = (+1) + (-2)
    assert by_key[k_split].next_keys == (k13, k18)    # 후계자 2개
    assert by_key[k_split].reward == 0.0
    assert by_key[k_split].terminal is False

    assert by_key[k13].mc_return == 1.0
    assert by_key[k13].reward == 1.0
    assert by_key[k13].next_keys == ()
    assert by_key[k13].terminal is True

    assert by_key[k18].mc_return == -2.0
    assert by_key[k18].reward == -2.0
    assert by_key[k18].terminal is True


def test_next_keys_길이는_종단0_히트1_스플릿2():
    k12 = StateKey(12, 0, 10, 1, 0, 0)
    k17 = StateKey(17, 0, 10, 0, 0, 0)
    rec = HandRecord(
        trajectory=[(k12, HIT), (k17, STAND)],
        parent=None, parent_step=None, bet=1.0, result=-1.0, subtree_result=-1.0,
    )
    r = RoundResult(
        hands=[rec], dealer_up=10, dealer_cards=[10, 8],
        net=-1.0, n_decisions=2, tc_at_deal=0.0,
    )
    trs = compute_transitions(r)

    assert len(trs) == 2
    assert len(trs[0].next_keys) == 1                 # 히트 → 후계자 1개
    assert trs[0].next_keys == (k17,)
    assert trs[0].reward == 0.0                       # 비종단은 보상 0
    assert trs[0].mc_return == -1.0                   # 이 손 자신의 결과
    assert trs[0].terminal is False
    assert len(trs[1].next_keys) == 0                 # 종단 → 후계자 0개
    assert trs[1].reward == -1.0
    assert trs[1].terminal is True


def test_결정없이_끝난_자식의_결과는_즉시보상으로_들어간다():
    """에이스 스플릿 자식은 한 장 받고 끝나 결정이 없다.
    부트스트랩할 다음 상태가 없으므로 그 결과는 reward에 들어가야 한다."""
    k_split = StateKey(12, 1, 6, 1, 1, 0)
    분기 = HandRecord(
        trajectory=[(k_split, SPLIT)],
        parent=None, parent_step=None, bet=1.0, result=0.0, subtree_result=2.0,
    )
    자식1 = HandRecord(trajectory=[], parent=0, parent_step=0, bet=1.0,
                       result=1.0, subtree_result=1.0)
    자식2 = HandRecord(trajectory=[], parent=0, parent_step=0, bet=1.0,
                       result=1.0, subtree_result=1.0)
    r = RoundResult(
        hands=[분기, 자식1, 자식2], dealer_up=6, dealer_cards=[6, 10, 10],
        net=2.0, n_decisions=1, tc_at_deal=0.0,
    )

    assert_invariants(r)
    trs = compute_transitions(r)
    assert len(trs) == 1
    assert trs[0].next_keys == ()
    assert trs[0].reward == 2.0
    assert trs[0].mc_return == 2.0
    assert trs[0].terminal is False


def test_불변식이_깨지면_예외가_난다():
    k = StateKey(20, 0, 6, 1, 0, 0)
    rec = HandRecord(trajectory=[(k, STAND)], parent=None, parent_step=None,
                     bet=1.0, result=1.0, subtree_result=1.0)
    r = RoundResult(hands=[rec], dealer_up=6, dealer_cards=[6, 10, 10],
                    net=99.0, n_decisions=1, tc_at_deal=0.0)
    try:
        assert_invariants(r)
    except AssertionError:
        return
    raise AssertionError("net이 어긋났는데 assert_invariants가 통과했다")


def test_Transition은_네임드튜플이라_필드_여섯_개다():
    k = StateKey(20, 0, 6, 1, 0, 0)
    tr = Transition(k, STAND, 1.0, (), 1.0, True)
    assert len(tr) == 6
    assert tr.key == k
    assert tr.action == STAND
