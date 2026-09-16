"""이 파일은 스플릿한 손의 카드 복원과 손별 실제 스플릿 깊이를 확인한다.
입력: 시드와 행동 목록(스플릿 우선 정책 포함).
출력: 스플릿 카드 복원·실제 깊이·상태 키 깊이에 대한 pytest 결과.
"""

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.env import BlackjackEnv  # noqa: E402
from blackjack_rl.hand import Hand  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.state import SPLIT, STAND  # noqa: E402
from game.engine import Finished, Pending, play  # noqa: E402

# 왜 이 시드인가: 누락 탐색이 찾은 재현 시드다. 2,2 대 6에서 두 번 갈라져
#     손 3이 실제 깊이 2가 되고, 깊이 0 Q와 깊이 2 Q의 손실이 뚜렷이 다르다.
재현시드 = 351727117773760389


# 왜 아래 두 도우미를 test_game_engine.py와 따로 두는가: tests/는 패키지가 아니라
#     테스트 모듈끼리 import하면 실행 방식에 따라 깨진다. 몇 줄이라 복사가 낫다.
def 손합계(카드들):
    """카드 목록의 블랙잭 합계와 소프트 여부."""
    합 = sum(11 if c == 1 else c for c in 카드들)
    에이스 = sum(1 for c in 카드들 if c == 1)
    while 합 > 21 and 에이스:
        합 -= 10
        에이스 -= 1
    return 합, (1 if (에이스 > 0 and 합 <= 21) else 0)


def 끝까지(seed, 고르기):
    """한 라운드를 끝까지 진행하고 (대기상태들, 결과)를 돌려준다."""
    행동, 본것 = [], []
    보기 = play(seed, 행동)
    while isinstance(보기, Pending):
        본것.append(보기)
        행동 = [*행동, 고르기(보기)]
        보기 = play(seed, 행동)
    return 본것, 보기


def test_스플릿한_뒤에도_카드_복원이_맞는다():
    난수 = np.random.default_rng(7)
    스플릿본판, 손세개이상 = 0, 0
    for seed in range(1500):
        본것, _결과 = 끝까지(
            seed, lambda v: SPLIT if SPLIT in v.legal else int(난수.choice(v.legal)))
        # 왜 n_hands로 보는가: RULES_V1은 include_split_depth_in_state=False라
        #     key.split_depth가 언제나 0이다. 실측으로 확인했다.
        손개수 = max((v.n_hands for v in 본것), default=1)
        if 손개수 > 1:
            스플릿본판 += 1
            for v in 본것:
                assert 손합계(v.player_cards) == (v.key.total, v.key.is_soft), (
                    f"seed={seed} 카드={v.player_cards} 키={v.key}")
        if 손개수 > 2:
            손세개이상 += 1
    # 왜 이 하한인가: 스플릿 우선 5,000판에서 648판이 갈라졌다(13.0%).
    #     1,500판이면 100판은 넉넉히 넘는다.
    assert 스플릿본판 >= 100, f"스플릿이 난 판이 {스플릿본판}뿐이다"
    # 왜 이것도 보는가: 재스플릿(손 3개 이상)에서 자식 인덱스 예측이 틀리면
    #     엉뚱한 손에 카드를 붙이게 된다. 5,000판에서 260판이 여기까지 갔다.
    assert 손세개이상 >= 20, f"손이 3개 이상 된 판이 {손세개이상}뿐이다"


def test_에이스_스플릿_자식은_두_장이다():
    시드들 = [s for s in range(3000)
            if isinstance(v := play(s, []), Pending) and v.player_cards == (1, 1)]
    assert len(시드들) >= 5, f"에이스 페어 시드가 {len(시드들)}개뿐이다"
    for s in 시드들:
        assert SPLIT in play(s, []).legal
        결과 = play(s, [SPLIT])
        # 왜 곧바로 끝나는가: 에이스 스플릿 자식은 한 장만 받고 결정 없이 끝난다.
        assert isinstance(결과, Finished)
        assert len(결과.hands) == 2
        for h in 결과.hands:
            assert len(h.cards) == 2 and h.cards[0] == 1, f"seed={s} {h}"
            assert h.total == 손합계(h.cards)[0]


def test_스플릿한_손의_실제_깊이를_내보낸다():
    # 왜: 키의 깊이는 규칙상 0이라, 그것으로 DP를 찾으면 깊이 0 Q로 손실을 잰다.
    #     엔진이 손마다 실제 깊이를 따로 내보내야 API가 맞는 Q를 고를 수 있다.
    처음 = play(재현시드, [])
    assert isinstance(처음, Pending)
    assert (처음.player_cards, 처음.dealer_up) == ((2, 2), 6)
    assert (처음.hand_index, 처음.split_depth) == (0, 0)
    # 손 0 → 자식 1·2(깊이 1), 손 1 → 자식 3·4(깊이 2). 손 2는 스탠드한다.
    기대 = {(SPLIT,): (1, 1), (SPLIT, SPLIT): (2, 1),
            (SPLIT, SPLIT, STAND): (3, 2), (SPLIT, SPLIT, STAND, STAND): (4, 2)}
    for 행동, (자리, 깊이) in 기대.items():
        보기 = play(재현시드, list(행동))
        assert isinstance(보기, Pending), 행동
        assert (보기.hand_index, 보기.split_depth) == (자리, 깊이), 행동
        assert 보기.key.split_depth == 0, 행동
    끝 = play(재현시드, [SPLIT, SPLIT, STAND, STAND, STAND])
    assert isinstance(끝, Finished)
    # 왜 손이 셋인가: 갈라진 손 0·1은 화면에서 빠지고 손 2(깊이 1)·3·4(깊이 2)만 남는다.
    assert [h.split_depth for h in 끝.hands] == [1, 2, 2]


@pytest.fixture
def 코어깊이(monkeypatch):
    """코어 Hand가 상태 키를 만들 때와 정산할 때의 split_depth를 남긴다."""
    기록 = {"키": [], "정산": []}
    원래키 = Hand.state_key
    원래정산 = BlackjackEnv._settle

    def 키감싼(self, dealer_up, rules, hands_in_round):
        기록["키"].append(self.split_depth)
        return 원래키(self, dealer_up, rules, hands_in_round)

    def 정산감싼(self, hands, records, dealer_total, dealer_bj):
        원래정산(self, hands, records, dealer_total, dealer_bj)
        # 왜 분기 노드를 빼는가: Finished.hands도 SPLIT으로 끝난 손을 보여 주지 않는다.
        기록["정산"].append([h.split_depth for h, r in zip(hands, records)
                          if not (r.trajectory and r.trajectory[-1][1] == SPLIT)])

    monkeypatch.setattr(Hand, "state_key", 키감싼)
    monkeypatch.setattr(BlackjackEnv, "_settle", 정산감싼)
    return 기록


def test_깊이는_최대_스플릿_깊이를_넘지_않는다(코어깊이):
    # 왜 코어와 대는가: 엔진은 자식이 목록 끝에 붙는다는 가정으로 깊이를 예측한다.
    #     코어 Hand.split_depth가 정답이므로 대기상태·끝난 손 모두 그것과 같아야 한다.
    난수 = np.random.default_rng(5)
    최대 = RULES_V1.max_split_depth
    깊이별 = Counter()
    for seed in range(2000):
        행동 = []
        while True:
            코어깊이["키"].clear()
            코어깊이["정산"].clear()
            보기 = play(seed, 행동)
            if not isinstance(보기, Pending):
                break
            assert 0 <= 보기.split_depth <= 최대, f"seed={seed} {보기}"
            assert 보기.split_depth == 코어깊이["키"][-1], f"seed={seed} {보기}"
            if 보기.split_depth >= 최대:
                assert SPLIT not in 보기.legal, f"seed={seed} {보기}"
            깊이별[보기.split_depth] += 1
            행동.append(SPLIT if SPLIT in 보기.legal else int(난수.choice(보기.legal)))
        보인깊이 = [h.split_depth for h in 보기.hands]
        assert 보인깊이 == 코어깊이["정산"][0], f"seed={seed} 보인={보인깊이}"
        assert all(0 <= d <= 최대 for d in 보인깊이), f"seed={seed} {보인깊이}"
    # 왜 이 하한인가: 깊은 손이 실제로 나와야 대조가 뜻을 갖는다. 이 2,000판
    #     실측(깊이 2 대기 292개, 깊이 3 대기 83개)의 절반쯤으로 잡았다.
    assert 깊이별[2] >= 145 and 깊이별[3] >= 40, dict(깊이별)


def test_상태_키의_깊이는_규칙대로_0이다():
    # 왜: decisions의 6필드는 학습·DP 표의 키 정의 그대로여야 한다. RULES_V1은
    #     키에서 깊이를 빼므로 실제 깊이가 얼마든 key.split_depth는 0이어야 한다.
    assert RULES_V1.include_split_depth_in_state is False
    깊이켠규칙 = replace(RULES_V1, include_split_depth_in_state=True)
    난수 = np.random.default_rng(9)
    깊은대기 = 0
    for seed in range(1000):
        행동, 보기 = [], play(seed, [])
        while isinstance(보기, Pending):
            assert 보기.key.split_depth == 0, f"seed={seed} {보기}"
            # 왜 깊이를 켠 규칙으로도 재생하는가: 그 규칙에서는 키가 곧 코어의 실제 깊이다.
            #     나머지 키 칸은 같고 깊이 칸만 Pending.split_depth와 같아야 한다.
            켠보기 = play(seed, 행동, rules=깊이켠규칙)
            assert 켠보기.key == 보기.key._replace(split_depth=보기.split_depth)
            assert 켠보기.split_depth == 보기.split_depth
            깊은대기 += 보기.split_depth >= 1
            행동.append(SPLIT if SPLIT in 보기.legal else int(난수.choice(보기.legal)))
            보기 = play(seed, 행동)
    # 왜 이 하한인가: 깊이 1 이상인 대기상태가 없으면 위 단언이 0만 보고 통과한다.
    #     이 1,000판 실측 533개의 절반쯤으로 잡았다.
    assert 깊은대기 >= 250, f"깊이 1 이상 대기상태가 {깊은대기}개뿐이다"
