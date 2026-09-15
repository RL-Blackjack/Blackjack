"""이 파일은 runner.run이 프레임 배열 13개와 meta를 올바르게 채우는지 검증한다
입력: 5,000 에피소드짜리 작은 ExperimentConfig
출력: pytest 통과/실패
"""

import numpy as np
import pytest

from blackjack_rl.rules import RULES_H17, RULES_V1
from blackjack_rl.starts import ExploringStarts, NaturalDeal
from blackjack_rl.state import Q_SHAPE, REACHABLE_KEYS
from blackjack_rl.train.config import ExperimentConfig
from blackjack_rl.train.runner import _make_sampler, run

# DP 정확 하우스엣지. 학습 정책이 이보다 좋게 나오면 무조건 버그다(설계서 §8.2).
DP_최적_EV = -0.005108


def 작은설정(**바꿀것) -> ExperimentConfig:
    바탕 = dict(name="mc_smoke", algo="mc", rules=RULES_V1,
               seed=42, n_episodes=5_000, n_frames=20)
    바탕.update(바꿀것)
    return ExperimentConfig(**바탕)


def test_산출물_배열_열세_개의_모양과_dtype이_설계서_표와_같다():
    art = run(작은설정())
    F = 20
    assert art.episodes.shape == (F,) and art.episodes.dtype == np.int64
    assert art.policy_full.shape == (F, len(REACHABLE_KEYS))
    assert art.policy_full.dtype == np.int8
    assert art.chart_action.shape == (F, 36, 10) and art.chart_action.dtype == np.int8
    assert art.chart_notation.shape == (F, 36, 10)
    assert art.chart_notation.dtype == np.dtype("<U2")
    assert art.chart_margin.shape == (F, 36, 10)
    assert art.chart_margin.dtype == np.float32
    assert art.chart_visits.shape == (F, 36, 10)
    assert art.chart_visits.dtype == np.int32
    for 이름 in ("ev_greedy", "ev_behavior", "agree_a", "agree_b", "maxq_minus_vstar"):
        배열 = getattr(art, 이름)
        assert 배열.shape == (F,), 이름
        assert 배열.dtype == np.float32, 이름
    assert art.q_final.shape == Q_SHAPE and art.q_final.dtype == np.float32
    assert art.n_final.shape == Q_SHAPE and art.n_final.dtype == np.int32


def test_프레임_에피소드는_1000에서_시작해_학습_끝에서_멈춘다():
    art = run(작은설정())
    assert art.episodes[0] == 1_000
    assert art.episodes[-1] == 5_000
    assert np.all(np.diff(art.episodes) > 0)


def test_정책_배열은_0에서_3_사이_행동_번호만_담는다():
    art = run(작은설정())
    assert art.policy_full.min() >= 0
    assert art.policy_full.max() <= 3


def test_학습_정책의_정확_EV는_어떤_프레임에서도_DP_최적보다_나쁘다():
    # 왜: 블랙잭 RL의 버그는 대부분 'EV가 좋아지는' 방향으로 나타난다
    #     (평가 시 탐험 잔존, 스플릿 이중계산, 더블 배당 스케일 오류).
    #     이 한 줄이 그 전부를 잡는 가장 싼 경보기다.
    art = run(작은설정())
    assert np.all(art.ev_greedy < DP_최적_EV)
    assert np.all(art.ev_greedy > -1.0)


def test_5천_에피소드_MC의_최종_EV는_측정된_구간_안에_있다():
    # 왜 이 구간인가: 같은 설정을 시드 1/42/7로 돌리면 최종 EV가
    #     -0.128543 / -0.116835 / -0.135420 이었다(실측). 하이퍼파라미터가 조금
    #     달라져도 흔들리지 않도록 아래를 -0.35로 넉넉히 잡되, 위쪽은 DP 최적값으로
    #     막아 '너무 잘하면 버그'를 그대로 유지한다.
    art = run(작은설정())
    assert -0.35 < float(art.ev_greedy[-1]) < DP_최적_EV


def test_행동정책_EV는_그리디_EV보다_항상_나쁘다():
    # 왜: eps=0.25면 네 행동 중 하나를 무작위로 고르는 비용이 붙는다.
    #     두 곡선이 겹치면 eps_greedy_dist나 evaluate_stochastic 배선이 틀린 것이다.
    art = run(작은설정())
    assert np.all(art.ev_behavior < art.ev_greedy)


def test_표_방문수는_프레임이_갈수록_줄지_않는다():
    art = run(작은설정())
    assert np.all(np.diff(art.chart_visits, axis=0) >= 0)
    assert art.chart_visits.min() >= 0
    assert art.chart_visits[-1].sum() > 0


def test_방문_0인_칸은_전략표_글자가_비어_있다():
    art = run(작은설정())
    마지막_방문 = art.chart_visits[-1]
    assert np.all(art.chart_action[-1][마지막_방문 == 0] == -1)
    assert np.all(art.chart_notation[-1][마지막_방문 == 0] == "")


def test_표기_배열이_D와_Ds를_구분해_남긴다():
    """저장된 npz만으로 GIF를 만들려면 이 구분이 프레임 배열에 남아 있어야 한다."""
    from blackjack_rl.chartspec import CHART_ROWS

    art = run(작은설정())
    표기 = set(np.unique(art.chart_notation))
    assert 표기 <= {"", "S", "H", "D", "Ds", "Y", "N"}

    # 액션 2(DOUBLE)인 칸은 표기가 D 아니면 Ds다 — 단 **페어 행은 제외**한다.
    # 왜: 페어 행은 행동과 무관하게 Y(쪼갠다)/N(안 쪼갠다)만 쓴다. 대표적으로
    #     5,5는 하드 10이라 딜러 2~9에 '쪼개지 말고 더블'이 정답이므로
    #     표기 N + 액션 DOUBLE이 정상이다. DP 최적표에도 그런 칸이 8개 있다.
    페어행 = np.array([row.kind == "pair" for row in CHART_ROWS])
    비페어 = ~페어행[None, :, None]              # [1, 36, 1] -> 프레임 전체에 방송
    더블칸 = (art.chart_action == 2) & 비페어
    assert set(np.unique(art.chart_notation[더블칸])) <= {"D", "Ds"}

    # 페어 행에서 액션이 DOUBLE인 칸은 반드시 N이다(쪼갰다면 액션이 SPLIT이어야 한다).
    페어더블 = (art.chart_action == 2) & ~비페어
    if 페어더블.any():
        assert set(np.unique(art.chart_notation[페어더블])) == {"N"}


def test_편향_곡선은_프로브_50칸으로_잰_유한한_값이다():
    art = run(작은설정())
    assert np.all(np.isfinite(art.maxq_minus_vstar))
    # 왜 1.0인가: 프로브 50칸의 V* 평균이 -0.3079509031이고 Q는 [-2, 1.5] 안에 있다.
    #     610칸 전체 평균(V* +0.0906)으로 잘못 계산하면 초기 프레임에서 이 경계를 넘는다.
    assert np.all(np.abs(art.maxq_minus_vstar) < 1.0)


def test_일치율_훅이_없으면_nan으로_남는다():
    art = run(작은설정())
    assert np.all(np.isnan(art.agree_a))
    assert np.all(np.isnan(art.agree_b))


def test_일치율_훅을_꽂으면_프레임마다_불린다():
    호출 = []

    def 가짜_일치율(표, 방문):
        호출.append((표.notation.shape, 방문.shape))
        return 0.5, 0.25

    art = run(작은설정(), agreement=가짜_일치율)
    assert len(호출) == 20
    assert 호출[0] == ((36, 10), (36, 10))
    assert np.allclose(art.agree_a, 0.5)
    assert np.allclose(art.agree_b, 0.25)


def test_진행_콜백은_프레임마다_한_번씩_불린다():
    본것 = []
    run(작은설정(), progress=lambda f, 전체, ep: 본것.append((f, 전체, ep)))
    assert len(본것) == 20
    assert 본것[0][0] == 1
    assert 본것[-1] == (20, 20, 5_000)


def test_시작_샘플러는_설정의_규칙을_그대로_쓴다():
    # 왜: ExploringStarts()를 인자 없이 만들면 기본값이 RULES_V1이라
    #     --rules h17 실행이 규칙이 어긋난 시작 쌍으로 학습하게 된다.
    assert isinstance(_make_sampler(작은설정()), NaturalDeal)
    샘플러 = _make_sampler(작은설정(start_dist="exploring", rules=RULES_H17))
    assert isinstance(샘플러, ExploringStarts)
    assert 샘플러.rules is RULES_H17


def test_doubleq는_두_표의_평균과_방문수_합을_저장한다():
    # 왜: agent.Q / agent.N을 그대로 저장하면 doubleq 실행의 표가 절반만 남아
    #     explain_state.py와 V3 화면이 반쪽짜리 Q를 보게 된다.
    art = run(작은설정(algo="doubleq", name="dq_smoke"))
    mc = run(작은설정())
    assert art.q_final.shape == Q_SHAPE
    assert not np.array_equal(art.q_final, mc.q_final)
    # 두 표를 합쳐 세므로 라운드당 결정 수가 mc와 같은 자릿수여야 한다.
    라운드당 = float(art.n_final.sum()) / 5_000
    assert 1.0 <= 라운드당 <= 3.0


def test_meta에_재현에_필요한_것이_전부_들어_있다():
    cfg = 작은설정()
    art = run(cfg)
    m = art.meta
    # 읽는 쪽 ①: scripts/evaluate.py 의 read_run_row 는 top-level을 본다.
    assert m["name"] == cfg.name
    assert m["algo"] == cfg.algo
    assert m["seed"] == cfg.seed
    assert m["n_episodes"] == 5_000
    assert m["wall_sec"] > 0.0
    assert m["rules_fp"] == RULES_V1.fingerprint()
    assert m["cfg_fp"] == cfg.fingerprint()
    # 읽는 쪽 ②: scripts/verify_release.py 의 config_from_meta 는 config 블록을 본다.
    for 키 in ("name", "algo", "seed", "n_episodes", "start_dist", "step",
              "alpha", "eps0", "eps_final", "eps_decay_at", "encoder", "n_frames"):
        assert 키 in m["config"], 키
    assert m["config"]["alpha"] == cfg.alpha
    assert m["seeds"]["master"] == 42
    assert set(m["seeds"]) >= {"master", "deal", "explore", "init", "eval"}
    assert isinstance(m["git_commit"], str)
    assert m["eps_per_sec"] > 0.0
    assert m["dp_ev_initial"] < 0.0
    # 왜: q_sha256은 save_run이 채운다. 여기서는 빈 칸으로 남아 있어야 맞다.
    assert m["q_sha256"] == ""


def test_같은_설정을_두_번_돌리면_Q가_비트까지_같다():
    a = run(작은설정())
    b = run(작은설정())
    assert np.array_equal(a.q_final, b.q_final)
    assert np.array_equal(a.n_final, b.n_final)
    assert np.array_equal(a.ev_greedy, b.ev_greedy)
