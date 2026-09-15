"""이 파일은 first-visit의 적용 범위가 '라운드'가 아니라 '손 궤적'인지 검사한다.
입력: 손으로 조립한 RoundResult.
출력: pytest 통과/실패."""

from blackjack_rl.env import HandRecord, RoundResult
from blackjack_rl.returns import compute_transitions
from blackjack_rl.state import HIT, SPLIT, STAND, StateKey


def test_스플릿한_두_손이_같은_키에_가면_표본이_두_개_나온다():
    """라운드 단위 first-visit로 잘못 구현하면 둘째 손 표본이 통째로 버려진다.
    발견하기 매우 어려운 버그라 여기서 못박는다."""
    k_split = StateKey(16, 0, 6, 1, 1, 0)
    k13 = StateKey(13, 0, 6, 1, 0, 0)

    분기 = HandRecord(trajectory=[(k_split, SPLIT)], parent=None, parent_step=None,
                      bet=1.0, result=0.0, subtree_result=0.0)
    첫손 = HandRecord(trajectory=[(k13, STAND)], parent=0, parent_step=0,
                      bet=1.0, result=1.0, subtree_result=1.0)
    둘째손 = HandRecord(trajectory=[(k13, STAND)], parent=0, parent_step=0,
                        bet=1.0, result=-1.0, subtree_result=-1.0)
    r = RoundResult(hands=[분기, 첫손, 둘째손], dealer_up=6, dealer_cards=[6, 10, 4],
                    net=0.0, n_decisions=3, tc_at_deal=0.0)

    trs = compute_transitions(r)
    표본 = [tr for tr in trs if tr.key == k13]
    assert len(표본) == 2
    리턴들 = sorted(tr.mc_return for tr in 표본)
    assert 리턴들 == [-1.0, 1.0]


def test_한_손_안에서는_첫_방문만_표본이_된다():
    """실제 플레이에서는 한 손 안에서 같은 키가 두 번 나올 수 없다(total이 단조 증가).
    그래도 first-visit 규칙이 코드에 실제로 존재하는지 인공 궤적으로 확인한다."""
    k = StateKey(12, 0, 10, 0, 0, 0)
    rec = HandRecord(trajectory=[(k, HIT), (k, STAND)], parent=None, parent_step=None,
                     bet=1.0, result=-1.0, subtree_result=-1.0)
    r = RoundResult(hands=[rec], dealer_up=10, dealer_cards=[10, 9],
                    net=-1.0, n_decisions=2, tc_at_deal=0.0)

    trs = compute_transitions(r)
    assert len(trs) == 1
    assert trs[0].action == HIT
