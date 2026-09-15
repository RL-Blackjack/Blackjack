"""이 파일은 모든 테스트가 함께 쓰는 픽스처를 둔다.
입력: 없음
출력: rules(정본 규칙), dp(정확 DP 해), repro_config/repro_run(재현성용 10만 에피소드 학습)
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


# 왜: 10만이라는 숫자는 실측에서 나왔다. 실제 MC 학습 루프가 59,919 eps/s이므로
#     10만 에피소드는 약 1.7초다. 재현성을 보이기에 충분히 길고, 기본 테스트
#     스위트(2분 예산)에 넣기에 충분히 짧은 유일한 자리다.
REPRO_EPISODES = 100_000


@pytest.fixture(scope="session")
def repro_config(rules):
    # 왜 함수 안에서 import하는가: conftest는 전체 스위트에 적용된다. 모듈 최상단에서
    #     blackjack_rl.train을 import하면 train/ 패키지가 아직 없는 태스크에서
    #     계획서 ①의 225개가 통째로 collection error가 난다. 지연 import면 이
    #     픽스처를 실제로 쓰는 테스트만 영향을 받는다.
    from blackjack_rl.train.config import ExperimentConfig

    return ExperimentConfig(
        name="repro",
        algo="mc",
        rules=rules,
        seed=20260916,
        n_episodes=REPRO_EPISODES,
        start_dist="natural",
        step="sample_average",
        alpha=0.02,
        eps0=0.25,
        eps_final=0.02,
        # 왜: 10만 에피소드짜리 실행에서 감쇠 지점이 500만이면 ε가 한 번도 안 내려간다.
        #     짧은 실행에서도 스케줄이 실제로 동작하도록 5만으로 당긴다.
        eps_decay_at=50_000,
        encoder="s1",
        # 왜: 프레임 하나당 DP 정책평가 2회(각 13.5ms + 14.9ms)가 붙는다.
        #     6프레임이면 0.2초라 재현성 확인에는 이걸로 충분하다.
        n_frames=6,
    )


@pytest.fixture(scope="session")
def repro_run(repro_config):
    # 왜 session 스코프인가: 이 파일의 테스트 다섯 개가 각자 학습을 다시 돌리면
    #   1.7초 × 5 = 8.5초가 된다. 결과가 불변이므로 한 번만 돌린다.
    from blackjack_rl.train.runner import run

    return run(repro_config)
