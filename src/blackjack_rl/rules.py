"""이 파일은 블랙잭 규칙 한 벌을 얼려서 보관하고 규칙마다 지문을 찍는다.
입력: 규칙 필드 값들(덱 방식, 딜러 S17/H17, 배당, 스플릿 상한 등).
출력: 불변 RuleSet 객체, 12자 지문 문자열, 한국어 설명 한 문단, 프리셋 4종.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, replace
from typing import Literal

DECK_MODES: tuple[str, str] = ("infinite", "shoe")


@dataclass(frozen=True)
class RuleSet:
    """게임 규칙 한 벌. 환경·RL·DP·웹앱이 전부 이 객체 하나를 본다."""

    deck_mode: Literal["infinite", "shoe"] = "infinite"
    decks: int = 4
    penetration: float = 0.75
    dealer_hits_soft_17: bool = False
    blackjack_payout: float = 1.5
    dealer_peeks: bool = True
    double_any_two: bool = True
    double_after_split: bool = True
    max_split_depth: int = 3
    max_hands_per_round: int = 4
    split_aces_one_card: bool = True
    resplit_aces: bool = False
    surrender: bool = False
    insurance: bool = False
    include_split_depth_in_state: bool = False

    def __post_init__(self) -> None:
        if self.deck_mode not in DECK_MODES:
            raise ValueError(
                f"deck_mode는 {DECK_MODES} 중 하나여야 한다: {self.deck_mode!r}"
            )

        if self.deck_mode == "infinite":
            # 왜: 무한덱에서 decks/penetration은 아무 의미가 없다. 값을 바꿔 두면
            #     "6덱으로 돌렸다"는 착각을 만들므로 기본값에서 벗어나면 바로 막는다.
            if self.decks != 4:
                raise ValueError(
                    "deck_mode='infinite'에서는 decks를 바꿀 수 없다. 기본값 4를 유지하라."
                )
            if self.penetration != 0.75:
                raise ValueError(
                    "deck_mode='infinite'에서는 penetration을 바꿀 수 없다. 기본값 0.75를 유지하라."
                )
        else:
            if not 1 <= self.decks <= 8:
                raise ValueError(f"decks는 1~8이어야 한다: {self.decks}")
            if not 0.1 <= self.penetration <= 1.0:
                raise ValueError(f"penetration은 0.1~1.0이어야 한다: {self.penetration}")

        if not 1.0 <= self.blackjack_payout <= 2.0:
            raise ValueError(
                f"blackjack_payout은 1.0~2.0이어야 한다(3:2면 1.5): {self.blackjack_payout}"
            )
        if not 0 <= self.max_split_depth <= 3:
            raise ValueError(f"max_split_depth는 0~3이어야 한다: {self.max_split_depth}")
        if not 1 <= self.max_hands_per_round <= 8:
            raise ValueError(
                f"max_hands_per_round는 1~8이어야 한다: {self.max_hands_per_round}"
            )
        if self.max_split_depth >= 1 and self.max_hands_per_round < 2:
            raise ValueError(
                "스플릿을 허용하면 라운드 손 수 상한이 최소 2여야 한다."
            )
        if self.resplit_aces and self.max_split_depth < 1:
            raise ValueError("resplit_aces=True인데 max_split_depth가 0이면 모순이다.")

        # 왜: 서렌더·인슈어런스는 설계서 §10에서 범위 밖으로 뺐다. 필드만 켜 두면
        #     환경이 조용히 무시해서 틀린 EV가 나오므로 만들기 자체를 막는다.
        if self.surrender:
            raise ValueError("surrender는 이번 학기 범위 밖이다. False로 두어라.")
        if self.insurance:
            raise ValueError("insurance는 이번 학기 범위 밖이다. False로 두어라.")

    def fingerprint(self) -> str:
        """규칙 한 벌을 12자 문자열로 요약한다. 파일명과 npz 메타에 박힌다."""
        # 왜: sha256은 실행할 때마다 값이 같다. 파이썬 내장 hash()는 실행마다 달라져
        #     파일명에 쓰면 어제 만든 표를 오늘 못 찾는다.
        조각들 = []
        for 필드 in dataclasses.fields(self):
            조각들.append(f"{필드.name}={getattr(self, 필드.name)!r}")
        본문 = ";".join(조각들)
        return hashlib.sha256(본문.encode("utf-8")).hexdigest()[:12]

    def describe_ko(self) -> str:
        """슬라이드에 그대로 붙일 수 있는 한국어 한 문단을 만든다."""
        if self.deck_mode == "infinite":
            덱설명 = "무한덱(복원추출)"
        else:
            덱설명 = f"{self.decks}덱 슈, 페네트레이션 {self.penetration:.0%}"

        if self.dealer_hits_soft_17:
            딜러설명 = "딜러는 소프트 17에서 히트한다(H17)"
        else:
            딜러설명 = "딜러는 소프트 17에서 스탠드한다(S17)"

        if self.blackjack_payout == 1.5:
            배당설명 = "내추럴 블랙잭 배당 3:2(1.5배)"
        else:
            배당설명 = f"내추럴 블랙잭 배당 {self.blackjack_payout}배"

        if self.dealer_peeks:
            피크설명 = "딜러는 업카드가 A나 10일 때 홀카드를 미리 확인한다(피크 있음)"
        else:
            피크설명 = "딜러는 홀카드를 미리 확인하지 않는다(피크 없음)"

        if self.double_any_two:
            더블설명 = "아무 두 장에서 더블 가능(DA2)"
        else:
            더블설명 = "합계 9~11에서만 더블 가능"

        if self.double_after_split:
            das설명 = "스플릿 후 더블 허용(DAS)"
        else:
            das설명 = "스플릿 후 더블 금지(NDAS)"

        if self.split_aces_one_card:
            에이스설명 = "에이스를 스플릿하면 각 손이 카드를 한 장만 받고 끝난다"
        else:
            에이스설명 = "에이스를 스플릿한 뒤에도 계속 카드를 받을 수 있다"

        if self.resplit_aces:
            리스플릿설명 = "에이스 리스플릿 허용"
        else:
            리스플릿설명 = "에이스 리스플릿 금지"

        상한설명 = (
            f"한 손은 최대 {self.max_split_depth}번까지 스플릿할 수 있고 "
            f"라운드 전체 손 수는 {self.max_hands_per_round}개를 넘지 않는다"
        )

        return (
            f"{덱설명}. {딜러설명}. {배당설명}. {피크설명}. "
            f"{더블설명}, {das설명}. {상한설명}. {에이스설명}, {리스플릿설명}. "
            f"항복과 보험은 쓰지 않는다. (규칙 지문 {self.fingerprint()})"
        )


RULES_V1: RuleSet = RuleSet()
RULES_H17: RuleSet = replace(RULES_V1, dealer_hits_soft_17=True)
RULES_NO_DAS: RuleSet = replace(RULES_V1, double_after_split=False)
RULES_SHOE: RuleSet = replace(RULES_V1, deck_mode="shoe")
