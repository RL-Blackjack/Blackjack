"""이 파일은 얼린 실험 설정과 12자 지문, 그리고 단일 CLI 파서를 검증한다
입력: ExperimentConfig / build_parser / config_from_args
출력: pytest 통과/실패
"""

import json

import pytest

from blackjack_rl.rng import make_streams
from blackjack_rl.rules import RULES_H17, RULES_V1
from blackjack_rl.train.config import (ALGOS, LIVE_EPS_PER_SEC, ExperimentConfig,
                                       build_parser, config_from_args)


def 기본설정(**바꿀것):
    바탕 = dict(name="mc_natural", algo="mc", rules=RULES_V1,
               seed=42, n_episodes=200_000)
    바탕.update(바꿀것)
    return ExperimentConfig(**바탕)


def test_설정은_얼어_있어서_고칠_수_없다():
    cfg = 기본설정()
    with pytest.raises(Exception):
        cfg.alpha = 0.5


def test_지문은_12자이고_두_번_만들어도_같다():
    a = 기본설정().fingerprint()
    b = 기본설정().fingerprint()
    assert len(a) == 12
    assert a == b
    assert all(문자 in "0123456789abcdef" for 문자 in a)


def test_이름만_다르면_지문이_같고_하이퍼파라미터가_다르면_달라진다():
    바탕 = 기본설정()
    assert 바탕.fingerprint() == 기본설정(name="다른이름").fingerprint()
    assert 바탕.fingerprint() != 기본설정(alpha=0.05).fingerprint()
    assert 바탕.fingerprint() != 기본설정(seed=43).fingerprint()
    assert 바탕.fingerprint() != 기본설정(rules=RULES_H17).fingerprint()


def test_엡실론은_선형으로_감쇠하고_바닥에서_멈춘다():
    cfg = 기본설정(eps0=0.25, eps_final=0.02, eps_decay_at=5_000_000)
    assert cfg.eps_at(0) == pytest.approx(0.25)
    assert cfg.eps_at(2_500_000) == pytest.approx(0.135)
    assert cfg.eps_at(5_000_000) == pytest.approx(0.02)
    assert cfg.eps_at(99_000_000) == pytest.approx(0.02)
    assert cfg.eps_at(-5) == pytest.approx(0.25)


def test_설정과_에이전트의_엡실론_식이_같다():
    """ε 스케줄이 두 군데 있으면 ev_behavior 곡선이 실제 행동정책과 어긋난다.
    runner는 cfg.eps_at(ep)로 행동정책을 만들고 에이전트는 자기 t로 ε를 쓴다.
    두 값이 같아야 곡선이 거짓말을 하지 않는다."""
    from blackjack_rl.agents.tabular import TabularAgent

    cfg = 기본설정(eps0=0.25, eps_final=0.02, eps_decay_at=1_000)
    agent = TabularAgent(RULES_V1, make_streams(0), algo="mc",
                         eps0=cfg.eps0, eps_final=cfg.eps_final,
                         eps_decay_at=cfg.eps_decay_at)
    for ep in (0, 1, 250, 500, 999, 1_000, 5_000):
        agent.t = ep
        assert agent.current_eps() == pytest.approx(cfg.eps_at(ep))


def test_json_덤프에_규칙과_두_지문이_전부_들어간다():
    cfg = 기본설정()
    본문 = json.loads(cfg.to_json())
    assert 본문["rules_fp"] == RULES_V1.fingerprint()
    assert 본문["cfg_fp"] == cfg.fingerprint()
    assert 본문["rules"]["dealer_hits_soft_17"] is False
    assert 본문["alpha"] == 0.02
    assert 본문["n_episodes"] == 200_000
    assert 본문["artifact_stem"] == cfg.artifact_stem()
    # 왜 이 키들을 확인하는가: scripts/verify_release.py의 config_from_meta가
    #   meta["config"] 하위에서 정확히 이 이름들을 읽어 설정을 되살린다.
    for 키 in ("name", "algo", "seed", "n_episodes", "start_dist", "step",
              "alpha", "eps0", "eps_final", "eps_decay_at", "encoder", "n_frames"):
        assert 키 in 본문


def test_파일이름_줄기는_이름_규칙지문_설정지문_시드_순서다():
    cfg = 기본설정()
    조각 = cfg.artifact_stem().split("__")
    assert len(조각) == 4
    assert 조각[0] == "mc_natural"
    assert 조각[1] == RULES_V1.fingerprint()
    assert 조각[2] == cfg.fingerprint()
    assert 조각[3] == "seed42"


def test_잘못된_값은_만들_때_바로_막힌다():
    with pytest.raises(ValueError):
        기본설정(algo="dqn")
    with pytest.raises(ValueError):
        기본설정(start_dist="무작위")
    with pytest.raises(ValueError):
        기본설정(step="adam")
    with pytest.raises(ValueError):
        # 왜: s2(트루카운트 포함)는 계획서 ④ W9의 몫이다.
        기본설정(encoder="s2")
    with pytest.raises(ValueError):
        기본설정(name="이름__에__밑줄두개")
    with pytest.raises(ValueError):
        기본설정(eps_final=0.9, eps0=0.2)
    with pytest.raises(ValueError):
        기본설정(alpha=0.0)
    with pytest.raises(ValueError):
        # n_frames=200이면 최소 1200 에피소드가 필요하다(로그 일정 시작점 1000 + 200)
        기본설정(n_episodes=1_199)


def test_알고리즘_목록에_dqn은_없다():
    # 왜: torch 의존 알고리즘은 이 패키지의 runner가 돌리지 않는다.
    assert "dqn" not in ALGOS
    assert ALGOS == ("mc", "q", "doubleq", "sarsa", "esarsa")


def test_CLI_플래그가_그대로_설정이_된다():
    args = build_parser().parse_args([
        "--name", "q_exploring", "--algo", "q", "--rules", "h17",
        "--seed", "7", "--episodes", "50000", "--start-dist", "exploring",
        "--step", "constant", "--alpha", "0.05", "--frames", "50",
    ])
    cfg = config_from_args(args)
    assert cfg.name == "q_exploring"
    assert cfg.algo == "q"
    assert cfg.rules is RULES_H17
    assert cfg.seed == 7
    assert cfg.n_episodes == 50_000
    assert cfg.start_dist == "exploring"
    assert cfg.step == "constant"
    assert cfg.alpha == 0.05
    assert cfg.n_frames == 50


def test_라이브는_초를_에피소드_수로_환산한다():
    args = build_parser().parse_args(
        ["--live", "--live-seconds", "10", "--live-every", "5"])
    cfg = config_from_args(args)
    # 왜: 실제 MC 학습 루프를 6만 에피소드 돌려 59,919 eps/s가 나왔다.
    #     그 실측값을 6만으로 반올림해 LIVE_EPS_PER_SEC에 박았다.
    assert LIVE_EPS_PER_SEC == 60_000
    assert cfg.n_episodes == 10 * LIVE_EPS_PER_SEC
    assert cfg.n_frames == 2


def test_산출물_폴더_플래그는_artifacts_하나뿐이다():
    """--out-dir과 --artifacts가 둘 다 있으면 어느 쪽이 이기는지 아무도 모른다."""
    parser = build_parser()
    이름들 = {행동.dest for 행동 in parser._actions}
    assert "artifacts" in 이름들
    assert "out_dir" not in 이름들
    assert "start_dist" in 이름들
    assert "start" not in 이름들
    # 일치율 훅의 표본 수는 하이퍼파라미터가 아니라 보고 설정이므로 설정 지문에 안 들어간다.
    assert "agreement_hands" in 이름들
