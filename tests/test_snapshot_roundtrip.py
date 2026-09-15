"""이 파일은 학습 산출물 npz의 저장·되읽기와 로그 간격 프레임 일정을 검증한다
입력: 난수로 채운 RunArtifact와 tmp_path
출력: pytest 통과/실패
"""

import hashlib

import numpy as np
import pytest

from blackjack_rl.rng import STREAM_NAMES, make_streams
from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import Q_SHAPE, REACHABLE_KEYS
from blackjack_rl.train.config import ExperimentConfig
from blackjack_rl.train.snapshot import (ARRAY_FIELDS, RunArtifact, artifact_path,
                                         config_path, git_commit_sha, load_run,
                                         log_schedule, q_sha256, save_run,
                                         seed_record)

NOTATIONS = np.array(["", "S", "H", "D", "Ds", "Y", "N"], dtype="<U2")


def 가짜_산출물(frames: int = 200) -> RunArtifact:
    """실제 학습과 같은 모양·dtype을 가진 난수 산출물. 압축이 가장 안 되는 최악 경우다."""
    rng = np.random.default_rng(0)
    return RunArtifact(
        episodes=log_schedule(200_000, frames),
        policy_full=rng.integers(0, 4, (frames, len(REACHABLE_KEYS))).astype(np.int8),
        chart_action=rng.integers(-1, 4, (frames, 36, 10)).astype(np.int8),
        chart_notation=NOTATIONS[rng.integers(0, len(NOTATIONS), (frames, 36, 10))],
        chart_margin=rng.standard_normal((frames, 36, 10)).astype(np.float32),
        chart_visits=rng.integers(0, 10**6, (frames, 36, 10)).astype(np.int32),
        ev_greedy=rng.standard_normal(frames).astype(np.float32),
        ev_behavior=rng.standard_normal(frames).astype(np.float32),
        agree_a=rng.standard_normal(frames).astype(np.float32),
        agree_b=rng.standard_normal(frames).astype(np.float32),
        maxq_minus_vstar=rng.standard_normal(frames).astype(np.float32),
        q_final=rng.standard_normal(Q_SHAPE).astype(np.float32),
        n_final=rng.integers(0, 10**6, Q_SHAPE).astype(np.int32),
        meta={"name": "가짜", "n_episodes": 200_000, "q_sha256": ""},
    )


def test_배열_키는_열세_개다():
    assert len(ARRAY_FIELDS) == 13
    assert "chart_notation" in ARRAY_FIELDS


def test_로그일정은_1000에서_시작해_정확히_n에서_끝난다():
    일정 = log_schedule(200_000, 200)
    assert 일정.shape == (200,)
    assert 일정.dtype == np.int64
    assert 일정[0] == 1_000
    assert 일정[-1] == 200_000


@pytest.mark.parametrize("n", [1_200, 2_000, 10_000, 200_000, 10_000_000])
def test_로그일정은_어떤_규모에서도_순증가한다(n):
    # 왜: 앞쪽 로그 간격이 1보다 좁으면 반올림 때문에 같은 에피소드가 두 번 나온다.
    #     프레임이 중복되면 학습곡선 x축이 멈춰 보인다.
    일정 = log_schedule(n, 200)
    assert np.all(np.diff(일정) > 0)
    assert int(일정.max()) <= n


def test_에피소드가_너무_적으면_일정을_만들_수_없다():
    with pytest.raises(ValueError):
        log_schedule(1_199, 200)


def test_저장했다_읽으면_배열_열세_개가_그대로다(tmp_path):
    원본 = 가짜_산출물(frames=8)
    경로 = tmp_path / "t.npz"
    save_run(경로, 원본)
    되읽음 = load_run(경로)
    for 이름 in ARRAY_FIELDS:
        기대 = getattr(원본, 이름)
        실제 = getattr(되읽음, 이름)
        assert 실제.dtype == 기대.dtype, 이름
        assert np.array_equal(실제, 기대), 이름
    assert 되읽음.meta["name"] == "가짜"
    assert 되읽음.meta["n_episodes"] == 200_000


def test_표기_배열은_U2로_왕복하고_Ds가_살아남는다(tmp_path):
    """chart_action만으로는 D와 Ds가 둘 다 action=2라 구분이 안 된다.
    저장된 npz만으로 GIF를 만들려면 이 배열이 반드시 있어야 한다."""
    원본 = 가짜_산출물(frames=4)
    원본.chart_notation[0, 0, 0] = "Ds"
    원본.chart_notation[0, 0, 1] = "D"
    경로 = tmp_path / "t.npz"
    save_run(경로, 원본)
    되읽음 = load_run(경로)
    assert 되읽음.chart_notation.dtype == np.dtype("<U2")
    assert str(되읽음.chart_notation[0, 0, 0]) == "Ds"
    assert str(되읽음.chart_notation[0, 0, 1]) == "D"


def test_npz_키는_배열_열세_개와_meta_하나뿐이다(tmp_path):
    경로 = tmp_path / "t.npz"
    save_run(경로, 가짜_산출물(frames=4))
    with np.load(경로, allow_pickle=False) as z:
        assert sorted(z.files) == sorted([*ARRAY_FIELDS, "meta"])


def test_pickle_없이_읽힌다(tmp_path):
    # 왜: 설계서 §8.1의 'pickle 금지' 조항을 테스트로 박아 둔다.
    경로 = tmp_path / "t.npz"
    save_run(경로, 가짜_산출물(frames=4))
    with np.load(경로, allow_pickle=False) as z:
        assert z["q_final"].shape == Q_SHAPE


def test_저장은_q_final의_sha256을_돌려주고_meta에도_박는다(tmp_path):
    산출물 = 가짜_산출물(frames=4)
    경로 = tmp_path / "t.npz"
    해시 = save_run(경로, 산출물)
    손계산 = hashlib.sha256(
        np.ascontiguousarray(산출물.q_final, dtype=np.float32).tobytes()).hexdigest()
    assert 해시 == 손계산
    assert len(해시) == 64
    assert 산출물.meta["q_sha256"] == 해시
    assert load_run(경로).meta["q_sha256"] == 해시


def test_확장자가_npz가_아니면_거부한다(tmp_path):
    with pytest.raises(ValueError):
        save_run(tmp_path / "t.pkl", 가짜_산출물(frames=4))


def test_200프레임_산출물은_1MB_미만이다(tmp_path):
    # 왜: 설계서 §4.1이 Git LFS 없이 커밋 가능해야 한다고 못박았다.
    #     난수 데이터는 압축이 거의 안 되는 최악의 경우다.
    #     실측 압축 전 1,621,936 B → 압축 후 841,441 B (표기를 현실적으로 채우면 799,520 B).
    경로 = tmp_path / "t.npz"
    save_run(경로, 가짜_산출물(frames=200))
    assert 경로.stat().st_size < 1_048_576


def test_시드_기록에_스트림_네_개가_전부_들어간다():
    기록 = seed_record(make_streams(42), 42)
    assert 기록["master"] == 42
    for 이름 in STREAM_NAMES:
        assert 기록[이름]["entropy"] == 42
        assert isinstance(기록[이름]["spawn_key"], list)
    assert 기록["deal"]["spawn_key"] != 기록["eval"]["spawn_key"]


def test_git_커밋_sha는_문자열을_돌려준다():
    sha = git_commit_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_산출물_경로는_설정_지문을_파일명에_박는다(tmp_path):
    cfg = ExperimentConfig(name="mc", algo="mc", rules=RULES_V1,
                           seed=42, n_episodes=200_000)
    npz = artifact_path(cfg, tmp_path)
    js = config_path(cfg, tmp_path)
    assert npz.name == cfg.artifact_stem() + ".npz"
    assert js.name == cfg.artifact_stem() + ".json"
    assert cfg.fingerprint() in npz.name
