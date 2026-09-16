"""이 파일은 시드 재생 엔진이 결정론적이고 카드 복원이 정확한지 확인한다.
입력: 시드와 행동 목록.
출력: 결정론·불법행동 거부·카드 복원에 대한 pytest 결과.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.dp.exact import add_card  # noqa: E402
from blackjack_rl.env import BlackjackEnv, HandRecord, RoundResult  # noqa: E402
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND, StateKey  # noqa: E402
from game.engine import (  # noqa: E402
    Finished,
    IllegalAction,
    Pending,
    infer_card,
    new_seed,
    play,
    rebuild_hand_cards,
)


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


def test_전이에서_카드를_되찾는_것이_유일하다():
    # 왜: 카드 복원 전체가 이 유일성 위에 서 있다. 하나라도 겹치면 틀린 카드를
    #     보여 주게 된다. 도달 가능한 시작 상태 38개를 전부 확인한다.
    도메인 = ([(t, 0) for t in range(4, 22)] + [(t, 1) for t in range(12, 22)]
            + [(r, 0) for r in range(2, 11)] + [(11, 1)])
    assert len(도메인) == 38
    for 앞 in 도메인:
        결과 = {}
        for c in range(1, 11):
            결과.setdefault(add_card(앞[0], 앞[1], c), []).append(c)
        for 뒤, 카드들 in 결과.items():
            assert len(카드들) == 1, f"{앞} -> {뒤} 를 설명하는 카드가 {카드들}로 여럿이다"
            assert infer_card(앞, 뒤) == 카드들[0]


def test_설명할_수_없는_전이는_예외():
    with pytest.raises(ValueError):
        infer_card((10, 0), (10, 0))


def test_같은_시드와_같은_행동은_같은_상태를_만든다():
    for seed in range(20):
        a, b = play(seed, []), play(seed, [])
        assert type(a) is type(b)
        if isinstance(a, Pending):
            assert (a.key, a.player_cards, a.dealer_up) == (b.key, b.player_cards, b.dealer_up)
        else:
            assert (a.dealer_cards, a.net) == (b.dealer_cards, b.net)


def test_첫_대기상태는_플레이어_두_장과_딜러_업카드를_준다():
    보기 = next(v for v in (play(s, []) for s in range(50)) if isinstance(v, Pending))
    assert len(보기.player_cards) == 2
    assert 2 <= 보기.dealer_up <= 11
    assert 보기.key.dealer_up == 보기.dealer_up


def test_복원한_카드의_합계가_언제나_상태_키와_같다():
    # 왜: 카드 복원이 맞는지 보는 가장 강한 불변식이다. 계획을 쓰며 3,000판을
    #     돌려 3,580개 대기상태에서 불일치 0건임을 실측했다.
    난수 = np.random.default_rng(0)
    확인 = 0
    for seed in range(600):
        본것, _결과 = 끝까지(seed, lambda v: int(난수.choice(v.legal)))
        for v in 본것:
            assert 손합계(v.player_cards) == (v.key.total, v.key.is_soft), (
                f"seed={seed} 카드={v.player_cards} 키={v.key}")
            확인 += 1
    assert 확인 >= 500, f"확인한 대기상태가 너무 적다: {확인}"


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


def test_행동_목록이_DB_칸에_들어간다():
    # 왜 이 검사인가: rounds.actions 가 String(128)이다. 가장 결정을 많이 뽑아내는
    #     플레이(스플릿 > 히트 > 스탠드)로 눌러 보고 직렬화한 길이가 칸에 들어가는지
    #     본다. 실측 최댓값은 20,000판에서 16결정(문자열 32자)이었다.
    def 최악(v):
        if SPLIT in v.legal:
            return SPLIT
        return HIT if HIT in v.legal else STAND

    최대결정, 최대길이 = 0, 0
    for seed in range(4000):
        행동, 보기 = [], play(seed, [])
        while isinstance(보기, Pending):
            행동.append(최악(보기))
            보기 = play(seed, 행동)
            assert len(행동) <= 60, f"seed={seed}에서 결정이 60개를 넘었다"
        최대결정 = max(최대결정, len(행동))
        최대길이 = max(최대길이, len(",".join(str(a) for a in 행동)))

    assert 최대결정 >= 10, f"최악 플레이인데 결정이 {최대결정}뿐 — 탐색이 안 되고 있다"
    assert 최대길이 <= 128, f"직렬화 길이 {최대길이}가 actions 칸(128)을 넘는다"


def 스플릿못하는시드():
    """첫 결정에서 스플릿이 불법인 시드 하나를 찾는다."""
    for s in range(200):
        보기 = play(s, [])
        if isinstance(보기, Pending) and SPLIT not in 보기.legal:
            return s
    raise AssertionError("그런 시드가 200개 안에 없다")


def test_불법_행동은_거부된다():
    # 왜: 클라이언트가 아무 행동이나 보낼 수 있다. 서버가 막지 않으면 코어가
    #     ValueError로 죽고 500이 나간다.
    with pytest.raises(IllegalAction):
        play(스플릿못하는시드(), [SPLIT])


def test_범위_밖_행동_번호도_거부된다():
    시드 = next(s for s in range(50) if isinstance(play(s, []), Pending))
    for 나쁜값 in (-1, 4, 99):
        with pytest.raises(IllegalAction):
            play(시드, [나쁜값])


def test_스탠드하면_그_손은_끝난다():
    # 왜: 스플릿을 안 한 손에서 스탠드하면 더 물어볼 것이 없다.
    assert isinstance(play(스플릿못하는시드(), [STAND]), Finished)


def test_더블은_카드_한_장만_받고_끝난다():
    시드 = next(s for s in range(200)
              if isinstance(play(s, []), Pending) and DOUBLE in play(s, []).legal
              and SPLIT not in play(s, []).legal)
    assert isinstance(play(시드, [DOUBLE]), Finished)


def test_끝난_라운드는_딜러_카드와_순손익을_준다():
    난수 = np.random.default_rng(11)
    결과들 = []
    for seed in range(200):
        _본것, 결과 = 끝까지(seed, lambda v: int(난수.choice(v.legal)))
        assert isinstance(결과, Finished)
        assert len(결과.dealer_cards) >= 2
        assert 결과.hands, "손이 하나도 없다"
        결과들.append(결과.net)
    # 왜: 200판의 순손익이 전부 같으면 정산이 안 도는 것이다.
    assert len(set(결과들)) > 1


def test_바로_끝나는_라운드도_있다():
    # 왜: 플레이어나 딜러가 내추럴이면 결정이 하나도 없이 끝난다. 그때 서버가
    #     대기상태를 기대하고 있으면 터진다.
    즉시종료 = sum(1 for s in range(400) if isinstance(play(s, []), Finished))
    assert 즉시종료 > 0


def test_새_시드는_매번_다르다():
    assert len({new_seed() for _ in range(200)}) == 200


def test_시드가_커도_재현된다():
    큰시드 = 2**40 + 12345
    a, b = play(큰시드, []), play(큰시드, [])
    assert type(a) is type(b)


@pytest.fixture
def 코어손(monkeypatch):
    """BlackjackEnv._settle을 감싸 정산 직전 코어 손의 카드를 라운드마다 남긴다."""
    기록 = []
    원래 = BlackjackEnv._settle

    def 감싼(self, hands, records, dealer_total, dealer_bj):
        원래(self, hands, records, dealer_total, dealer_bj)
        # 왜 분기 노드를 빼는가: Finished.hands도 SPLIT으로 끝난 손을 보여 주지 않는다.
        기록.append([tuple(h.cards) for h, r in zip(hands, records)
                   if not (r.trajectory and r.trajectory[-1][1] == SPLIT)])

    monkeypatch.setattr(BlackjackEnv, "_settle", 감싼)
    return 기록


def test_끝난_손의_카드가_코어의_실제_손과_같다(코어손):
    # 왜: 마지막 히트·더블 카드, 에이스 스플릿 자식의 둘째 장, 내추럴 라운드의 두 장은
    #     act 콜백이 다시 불리지 않아 전이로 되찾을 수 없다. 코어가 정산에 쓴 카드와 직접 댄다.
    정책들 = [
        lambda v, r: int(r.choice(v.legal)),
        lambda v, r: SPLIT if SPLIT in v.legal else int(r.choice(v.legal)),
        lambda v, r: DOUBLE if DOUBLE in v.legal else int(r.choice(v.legal)),
    ]
    판수 = 에이스스플릿 = 더블 = 버스트 = 즉시 = 손셋이상 = 0
    for 번호, 고르기 in enumerate(정책들):
        난수 = np.random.default_rng(100 + 번호)
        for seed in range(번호 * 10_000, 번호 * 10_000 + 1000):
            코어손.clear()
            본것, 결과 = 끝까지(seed, lambda v: 고르기(v, 난수))
            assert len(코어손) == 1, f"seed={seed} 정산이 {len(코어손)}번 불렸다"
            assert [h.cards for h in 결과.hands] == 코어손[0], (
                f"seed={seed} 보인 손={[h.cards for h in 결과.hands]} 실제={코어손[0]}")
            for h in 결과.hands:
                assert h.total == 손합계(h.cards)[0], f"seed={seed} {h}"
                버스트 += h.total > 21
                더블 += h.bet == 2.0
            판수 += 1
            즉시 += 결과.n_decisions == 0
            손셋이상 += len(결과.hands) >= 3
            # 스플릿한 판의 손은 전부 자식이므로 첫 장이 에이스면 에이스 스플릿 자식이다.
            에이스스플릿 += sum(len(결과.hands) >= 2 and h.cards[0] == 1 for h in 결과.hands)
    assert 판수 >= 3000
    # 왜 이 하한들인가: 틀리던 네 경우(버스트, 더블, 내추럴, 에이스 스플릿)와 재스플릿이
    #     실제로 섞여 있어야 이 대조가 뜻을 갖는다. 실측(버스트 1,056손, 더블 1,620손,
    #     즉시 종료 246판, 에이스 스플릿 자식 12손, 손 3개 이상 50판)의 절반쯤으로 잡았다.
    assert 버스트 >= 500 and 더블 >= 800 and 즉시 >= 120, (버스트, 더블, 즉시)
    assert 에이스스플릿 >= 6 and 손셋이상 >= 25, (에이스스플릿, 손셋이상)


def test_버스트한_손은_합계가_21을_넘고_판돈을_잃는다():
    버스트 = 0
    for seed in range(600):
        _본것, 결과 = 끝까지(seed, lambda v: HIT if v.key.total < 17 else STAND)
        for h in 결과.hands:
            if h.total > 21:
                버스트 += 1
                assert h.result == -h.bet, f"seed={seed} {h}"
    # 왜 하한을 두는가: 예전 복원은 버스트를 부른 마지막 장을 빠뜨려 21을 넘는 합계를
    #     한 번도 보이지 않았다. 그러면 위 단언은 한 번도 돌지 않고 통과한다.
    #     600판 실측 173손의 절반쯤으로 잡았다.
    assert 버스트 >= 80, f"합계가 21을 넘는 손이 {버스트}개뿐이다"


def test_내추럴로_끝난_라운드도_두_장을_보여준다():
    즉시 = [v for v in (play(s, []) for s in range(400)) if isinstance(v, Finished)]
    assert len(즉시) >= 15, f"결정 없이 끝난 라운드가 {len(즉시)}개뿐이다"
    for v in 즉시:
        assert v.n_decisions == 0 and len(v.hands) == 1
        손 = v.hands[0]
        assert len(손.cards) == 2, f"손 0의 카드가 {손.cards}다"
        assert 손.total == 손합계(손.cards)[0]
        if 손.result > 0:
            # 왜: 결정 없이 이기는 길은 플레이어 블랙잭뿐이다.
            assert 손.total == 21 and sorted(손.cards) == [1, 10]


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


def test_뽑기_로그가_어긋나면_예외():
    키 = StateKey(total=11, is_soft=0, dealer_up=10, can_double=1, can_split=0)
    # 손 0: 5, 6 / 딜러 업·홀: 10, 6 / 손 0 히트: 4 / 딜러 추가: 3
    결과 = RoundResult(hands=[HandRecord(trajectory=[(키, HIT), (키, STAND)])],
                     dealer_up=10, dealer_cards=[10, 6, 3])
    정상 = [5, 6, 10, 6, 4, 3]
    assert rebuild_hand_cards(정상, 결과) == {0: [5, 6, 4]}
    for 망가진 in ([*정상, 7],              # 꼬리에 모르는 카드
                  [5, 6, 10, 6, 4, 9],     # 딜러 추가 카드가 다르다
                  [5, 6, 10, 6, 4],        # 딜러 추가 카드가 빠졌다
                  [5, 6, 10, 6]):          # 플레이어 몫까지 모자란다
        with pytest.raises(RuntimeError):
            rebuild_hand_cards(망가진, 결과)
