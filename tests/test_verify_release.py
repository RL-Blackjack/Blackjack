"""이 파일은 scripts/verify_release.py의 해시 재계산과 점검 항목을 검증한다
입력: 실제 runner가 쓰는 meta 모양을 그대로 흉내 낸 가짜 npz
출력: pytest 통과/실패
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_release  # noqa: E402

from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.state import LEGAL, Q_SHAPE  # noqa: E402
from blackjack_rl.train.config import ExperimentConfig  # noqa: E402


def 설정블록(**바꿀것) -> dict:
    """train/runner.py가 meta['config']에 넣는 것과 같은 모양."""
    cfg = ExperimentConfig(name="mc_natural", algo="mc", rules=RULES_V1,
                           seed=42, n_episodes=100_000, eps_decay_at=50_000,
                           n_frames=6)
    본문 = json.loads(cfg.to_json())
    본문.update(바꿀것)
    return 본문


def 가짜_스냅샷(path, *, ev_final=-0.008, sha_override=None, rules_fp=None):
    """실제 스냅샷과 같은 모양의 최소 npz를 만든다."""
    q = np.full(Q_SHAPE, -np.inf, dtype=np.float32)
    q[LEGAL] = 0.0
    참해시 = verify_release.sha256_of_array(q)
    설정 = 설정블록()
    meta = {
        "schema_version": 2,
        "name": 설정["name"], "algo": 설정["algo"], "seed": 설정["seed"],
        "n_episodes": 설정["n_episodes"], "n_frames": 설정["n_frames"],
        "rules_fp": rules_fp or RULES_V1.fingerprint(),
        "cfg_fp": 설정["cfg_fp"],
        "config": 설정,
        "wall_sec": 1.7,
        "q_sha256": sha_override or 참해시,
    }
    np.savez_compressed(
        path,
        q_final=q,
        n_final=np.zeros(Q_SHAPE, dtype=np.int32),
        ev_greedy=np.array([-0.30, ev_final], dtype=np.float32),
        episodes=np.array([1000, 100000], dtype=np.int64),
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    return 참해시


def 점검값(report, 이름조각):
    for 이름, 통과, _값 in report["checks"]:
        if 이름조각 in 이름:
            return 통과
    raise AssertionError(f"그런 점검 항목이 없다: {이름조각}")


def test_해시는_float32_바이트로_계산되어_재현된다():
    a = np.full(Q_SHAPE, -np.inf, dtype=np.float32)
    a[LEGAL] = 0.25
    b = a.astype(np.float64)
    # 왜: 같은 Q를 float64로 들고 있어도 같은 해시가 나와야 한다.
    #     안 그러면 dtype이 바뀔 때마다 가짜 실패가 난다.
    assert verify_release.sha256_of_array(a) == verify_release.sha256_of_array(b)
    assert len(verify_release.sha256_of_array(a)) == 64


def test_멀쩡한_스냅샷은_모든_점검을_통과한다(tmp_path):
    경로 = tmp_path / "good.npz"
    참해시 = 가짜_스냅샷(경로)
    보고 = verify_release.verify_npz(경로)
    assert 보고["ok"] is True
    assert 보고["q_sha256"] == 참해시
    assert 점검값(보고, "sha256") is True
    assert 점검값(보고, "NaN") is True
    assert 점검값(보고, "-inf") is True


def test_해시가_다르면_실패로_잡힌다(tmp_path):
    경로 = tmp_path / "bad_hash.npz"
    가짜_스냅샷(경로, sha_override="00" * 32)
    보고 = verify_release.verify_npz(경로)
    assert 보고["ok"] is False
    assert 점검값(보고, "sha256") is False


def test_EV가_DP_상한보다_좋으면_실패로_잡힌다(tmp_path):
    경로 = tmp_path / "too_good.npz"
    가짜_스냅샷(경로, ev_final=0.02)
    보고 = verify_release.verify_npz(경로)
    assert 보고["ok"] is False
    assert 점검값(보고, "DP 상한") is False
    assert 점검값(보고, "sanity") is False


def test_규칙_지문이_다르면_실패로_잡힌다(tmp_path):
    경로 = tmp_path / "wrong_rules.npz"
    가짜_스냅샷(경로, rules_fp="deadbeef1234")
    보고 = verify_release.verify_npz(경로)
    assert 보고["ok"] is False
    assert 점검값(보고, "규칙 지문") is False


def test_meta에서_설정을_되살린다(tmp_path):
    경로 = tmp_path / "good.npz"
    가짜_스냅샷(경로)
    with np.load(경로, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
    cfg = verify_release.config_from_meta(meta)
    assert cfg.seed == 42
    assert cfg.algo == "mc"
    assert cfg.n_episodes == 100_000
    assert cfg.rules == RULES_V1


def test_실제_runner가_만든_meta로도_설정이_그대로_되살아난다():
    """meta를 쓰는 쪽(runner)과 읽는 쪽(여기)이 갈라지면 실제 산출물에서만
    KeyError가 난다. 가짜 npz로는 끝까지 안 잡히므로 진짜 meta로 한 번 돈다."""
    from blackjack_rl.train.runner import run

    원본 = ExperimentConfig(name="repro_meta", algo="q", rules=RULES_V1,
                          seed=3, n_episodes=1_500, n_frames=4,
                          start_dist="exploring", step="constant", alpha=0.05)
    art = run(원본)
    되살린것 = verify_release.config_from_meta(art.meta)
    assert 되살린것 == 원본
    assert 되살린것.fingerprint() == art.meta["cfg_fp"]


def test_규칙이_바뀐_meta로는_설정을_되살리지_않는다():
    with pytest.raises(ValueError):
        verify_release.config_from_meta({"config": {"rules_fp": "deadbeef1234"}})


def test_보고서를_마크다운_표로_찍는다(tmp_path):
    경로 = tmp_path / "good.npz"
    가짜_스냅샷(경로)
    본문 = verify_release.render_report(verify_release.verify_npz(경로))
    assert "| 점검 항목 | 결과 | 값 |" in 본문
    assert "실패" not in 본문


def test_main은_실패하면_1을_돌려준다(tmp_path, capsys):
    경로 = tmp_path / "bad.npz"
    가짜_스냅샷(경로, sha_override="00" * 32)
    assert verify_release.main(["--run", str(경로)]) == 1
    assert "실패 있음" in capsys.readouterr().out
