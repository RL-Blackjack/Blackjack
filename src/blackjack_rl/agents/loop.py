"""이 파일은 다섯 알고리즘이 공유하는 단 하나의 학습 루프를 정의한다.
입력: 환경, 에이전트, 시작 상태 샘플러, 난수 생성기, 에피소드 수.
출력: 관측한 전이의 총 개수(에이전트의 Q는 제자리에서 갱신된다).
"""

from __future__ import annotations

from collections.abc import Callable

from blackjack_rl.cards import make_shoe
from blackjack_rl.env import BlackjackEnv
from blackjack_rl.returns import compute_transitions
from blackjack_rl.rng import Streams
from blackjack_rl.rules import RuleSet


def make_env(rules: RuleSet, streams: Streams) -> BlackjackEnv:
    """규칙이 정한 슈를 붙인 환경을 만든다."""
    # 왜: 카드는 반드시 deal 스트림에서만 나와야 한다. 여기 한 줄로 고정해 두면
    #     실험 스크립트마다 어떤 스트림을 넘겼는지 신경 쓸 일이 없다.
    shoe = make_shoe(rules, streams.deal)
    return BlackjackEnv(rules, shoe, streams)


def run_episodes(
    env: BlackjackEnv,
    agent,
    sampler,
    rng,
    n_episodes: int,
    progress: Callable[[int], None] | None = None,
    progress_every: int = 100_000,
    snapshot_at: set[int] | None = None,
    on_snapshot: Callable[[int], None] | None = None,
) -> int:
    """라운드를 n_episodes번 돌리며 에이전트를 학습시키고, 본 전이 수를 돌려준다.

    snapshot_at 에 든 에피소드 번호에서는 그 판의 학습이 끝난 직후 on_snapshot(ep)을
    부른다. train/runner.py 가 프레임을 찍는 유일한 방법이며, 덕분에 이 저장소에
    학습 루프는 이 함수 하나뿐이다.
    """
    if n_episodes < 0:
        raise ValueError(f"n_episodes는 0 이상이어야 한다: {n_episodes}")
    if progress_every < 1:
        raise ValueError(f"progress_every는 1 이상이어야 한다: {progress_every}")
    if snapshot_at is not None and on_snapshot is None:
        raise ValueError("snapshot_at을 줬으면 on_snapshot도 줘야 한다")

    n_transitions = 0
    for i in range(n_episodes):
        # 왜: NaturalDeal은 None을 돌려주고 환경이 알아서 딜한다. 그래서 이 루프에는
        #     '자연 딜이냐 탐험적 시작이냐' 분기가 한 줄도 없다. 시작 분포를 바꿔도
        #     학습 코드가 글자 하나 달라지지 않으므로 비교가 공정하다.
        start = sampler.sample(rng)
        result = env.play_round(agent.act, start=start)
        transitions = compute_transitions(result)
        agent.observe(transitions)
        n_transitions += len(transitions)

        ep = i + 1
        if snapshot_at is not None and ep in snapshot_at:
            # 왜 observe 다음인가: 스냅샷은 '그 에피소드까지 학습한 결과'여야 한다.
            #     앞에 두면 프레임 하나만큼 뒤처진 Q가 저장된다.
            on_snapshot(ep)
        if progress is not None and ep % progress_every == 0:
            progress(ep)

    return n_transitions
