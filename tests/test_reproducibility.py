"""이 파일은 같은 시드가 같은 결과를 내는지와 EV 경보가 살아 있는지를 검증한다
입력: conftest의 repro_config / repro_run 픽스처와 scripts/verify_release.py
출력: pytest 통과/실패
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_release  # noqa: E402

from blackjack_rl.eval.simulate import SanityAlarm, sanity_alarm  # noqa: E402
from blackjack_rl.state import LEGAL, Q_SHAPE  # noqa: E402
from blackjack_rl.train.runner import run  # noqa: E402
from blackjack_rl.train.snapshot import save_run  # noqa: E402

# 왜: 계획서 ①이 DP로 직접 푼 값이다. 학습이 이보다 좋으면 학습이 아니라 버그다.
DP_OPTIMAL_EV = -0.005108


def test_같은_시드로_다시_돌리면_q_final_해시가_같다(repro_config, repro_run):
    다시 = run(repro_config)
    첫해시 = verify_release.sha256_of_array(repro_run.q_final)
    둘해시 = verify_release.sha256_of_array(다시.q_final)
    assert 첫해시 == 둘해시, "같은 시드인데 q_final이 달라졌다 — 난수 스트림이 새고 있다"
    assert np.array_equal(repro_run.n_final, 다시.n_final)
    assert np.array_equal(repro_run.episodes, 다시.episodes)


def test_시드가_다르면_해시가_달라진다(repro_config, repro_run):
    # 왜: 위 테스트만 있으면 '항상 같은 상수를 돌려주는 버그'도 통과한다.
    #     시드를 바꿨을 때 실제로 달라지는지까지 봐야 의미가 있다.
    다른결과 = run(replace(repro_config, seed=repro_config.seed + 1))
    assert (verify_release.sha256_of_array(다른결과.q_final)
            != verify_release.sha256_of_array(repro_run.q_final))


def test_학습_뒤에도_불법_행동은_minus_inf다(repro_run):
    q = np.asarray(repro_run.q_final)
    assert q.shape == Q_SHAPE
    assert bool(np.isneginf(q[~LEGAL]).all())
    assert not bool(np.isnan(q).any())


def test_방문수_합이_라운드당_결정수와_맞는다(repro_run, repro_config):
    라운드당 = float(np.asarray(repro_run.n_final).sum()) / repro_config.n_episodes
    # 왜 이 구간인가: 그리디 정책으로 6만 라운드를 실측하니 라운드당 결정 수가
    #     1.24였고, 탐험적 시작 30만 판에서는 1.4425였다. ε 탐험이 섞이면 히트가
    #     늘어 더 커지므로 위쪽을 3.0까지 열어 둔다. 1.0 아래면 결정을 빠뜨린
    #     것이고 3.0 위면 전이를 이중으로 세고 있는 것이다.
    assert 1.0 <= 라운드당 <= 3.0, f"라운드당 결정 수가 {라운드당:.2f}다"


def test_학습_EV는_DP_상한을_절대_넘지_않는다(repro_run):
    최고 = float(np.asarray(repro_run.ev_greedy).max())
    # 왜: DP ev_initial은 모든 결정적 정책의 상한이다. 학습 정책이 이걸 넘으면
    #     평가에 학습 분포가 새어 들어갔거나 배당 계산이 틀린 것이다.
    #     ev_greedy가 float32로 저장되므로 1e-6만 여유를 준다.
    assert 최고 <= DP_OPTIMAL_EV + 1e-6, f"학습 EV {최고:+.6f}가 DP 상한을 넘었다"


def test_EV가_너무_좋으면_sanity_alarm이_터진다():
    # 통과해야 하는 쪽: 진짜 하우스엣지
    sanity_alarm(DP_OPTIMAL_EV)
    sanity_alarm(-0.05)
    # 터져야 하는 쪽: 블랙잭 RL의 버그는 대부분 결과가 '좋아지는' 방향으로 나타난다
    with pytest.raises(SanityAlarm):
        sanity_alarm(-0.001)
    with pytest.raises(SanityAlarm):
        sanity_alarm(0.02)


def test_저장한_스냅샷이_verify_release_점검을_통과한다(tmp_path, repro_run):
    경로 = tmp_path / "repro.npz"
    save_run(경로, repro_run)
    보고 = verify_release.verify_npz(경로)
    실패들 = [이름 for 이름, 통과, _값 in 보고["checks"] if not 통과]
    assert 보고["ok"] is True, f"실패한 점검: {실패들}"
