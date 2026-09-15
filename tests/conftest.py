"""이 파일은 모든 테스트가 함께 쓰는 픽스처를 둔다.
입력: 없음
출력: rules(정본 규칙), dp(정확 DP 해 - 세션당 한 번만 계산)
"""

import pytest

from blackjack_rl.dp.exact import solve_optimal
from blackjack_rl.rules import RULES_V1


@pytest.fixture(scope="session")
def rules():
    return RULES_V1


@pytest.fixture(scope="session")
def dp(rules):
    # 왜 session 스코프인가: solve_optimal은 수 초가 걸리는데 결과가 불변이다.
    #   테스트마다 다시 풀면 느려지기만 하고 얻는 게 없다.
    return solve_optimal(rules)
