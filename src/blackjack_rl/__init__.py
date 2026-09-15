"""이 파일은 blackjack_rl 패키지의 공개 API를 한곳에 모아 둔다.
입력: 없음(패키지를 import할 때 실행된다).
출력: 패키지 버전과 자주 쓰는 클래스·함수들의 재수출.
"""

from blackjack_rl.cards import (
    CARD_VALUES,
    CARDS_PER_DECK,
    FiniteShoe,
    InfiniteShoe,
    ReplayShoe,
    Shoe,
    card_value,
    hilo_tag,
    make_shoe,
)
from blackjack_rl.rng import Streams, make_streams
from blackjack_rl.rules import (
    RULES_H17,
    RULES_NO_DAS,
    RULES_SHOE,
    RULES_V1,
    RuleSet,
)

__version__: str = "0.1.0"

__all__ = [
    "CARDS_PER_DECK",
    "CARD_VALUES",
    "FiniteShoe",
    "InfiniteShoe",
    "RULES_H17",
    "RULES_NO_DAS",
    "RULES_SHOE",
    "RULES_V1",
    "ReplayShoe",
    "RuleSet",
    "Shoe",
    "Streams",
    "__version__",
    "card_value",
    "hilo_tag",
    "make_shoe",
    "make_streams",
]
