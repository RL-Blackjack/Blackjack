"""이 파일은 마스터 시드 하나를 독립된 난수 스트림 네 개로 갈라 준다.
입력: 마스터 시드 정수 하나.
출력: deal/explore/init/eval 네 개의 numpy Generator를 담은 Streams.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

STREAM_NAMES: tuple[str, str, str, str] = ("deal", "explore", "init", "eval")


class Streams(NamedTuple):
    """용도별로 갈라 놓은 난수 생성기 묶음."""

    deal: np.random.Generator      # 카드를 뽑을 때만
    explore: np.random.Generator   # ε-greedy 탐험에만
    init: np.random.Generator      # Q 테이블 초기 미세 노이즈에만
    eval: np.random.Generator      # 평가에만


def make_streams(master_seed: int) -> Streams:
    """마스터 시드에서 서로 겹치지 않는 Generator 네 개를 만든다."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, (int, np.integer)):
        raise TypeError(f"master_seed는 정수여야 한다: {master_seed!r}")
    if master_seed < 0:
        raise ValueError(f"master_seed는 0 이상이어야 한다: {master_seed}")

    # 왜: 시드 하나로 Generator 네 개를 따로 만들면(예: seed, seed+1, ...) 수열이
    #     겹칠 수 있다. spawn()은 겹치지 않는 자식 시드를 보장한다.
    # 왜: 용도별로 스트림을 나누면 알고리즘이 탐험 난수를 몇 번 쓰든 카드 순서가
    #     그대로다. "공정 비교를 어떻게 보장했나"에 대한 답이 이 세 줄이다.
    자식시드들 = np.random.SeedSequence(int(master_seed)).spawn(4)
    생성기들 = [np.random.default_rng(자식시드) for 자식시드 in 자식시드들]
    return Streams(*생성기들)
