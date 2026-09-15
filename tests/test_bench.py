"""이 파일은 scripts/bench.py의 측정 결과가 믿을 만한지 검증한다
입력: 없음 (bench 모듈을 직접 import해서 짧게 실행한다)
출력: pytest 통과/실패
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench  # noqa: E402


def test_두_마스크_방식의_반환값이_모든_키에서_같다():
    table = bench.build_legal_table()
    for key in bench.sample_keys():
        보는쪽 = bench.legal_by_table(table, key)
        만드는쪽 = bench.legal_by_build(key)
        assert np.array_equal(보는쪽, 만드는쪽), f"{key}에서 마스크가 다르다"
    assert bench.masks_are_equal() is True


def test_마스크는_스탠드와_히트를_항상_허용한다():
    table = bench.build_legal_table()
    for key in bench.sample_keys():
        mask = bench.legal_by_table(table, key)
        assert bool(mask[0]) is True
        assert bool(mask[1]) is True
        assert bool(mask[2]) == bool(key[3])
        assert bool(mask[3]) == bool(key[4])


def test_테이블_뷰는_새_배열을_만들지_않는다():
    table = bench.build_legal_table()
    view = bench.legal_by_table(table, (16, 0, 10, 1, 0, 0))
    assert view.base is not None


def test_라운드_시뮬레이션_속도가_양수다():
    result = bench.bench_rounds(2_000, seed=1, use_table=True)
    assert result["eps_per_sec"] > 0.0
    assert result["us_per_decision"] > 0.0
    assert result["decisions_per_round"] >= 1.0


def test_계획표_합계가_설계서의_6에서_8시간_구간에_들어온다():
    rows = bench.budget_rows(bench.PLANNED_EPS_PER_SEC)
    assert len(rows) == 8
    assert 6.0 <= bench.total_hours(rows) <= 8.5


def test_속도가_두_배면_예산_시간이_절반이_된다():
    느린쪽 = bench.total_hours(bench.budget_rows(10_000.0))
    빠른쪽 = bench.total_hours(bench.budget_rows(20_000.0))
    assert 빠른쪽 == pytest.approx(느린쪽 / 2.0)


def test_bench_json_스키마가_전부_채워진다(tmp_path):
    result = bench.run_bench(n_rounds=2_000, n_calls=5_000, seed=7)
    out = tmp_path / "bench.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))

    assert loaded["schema_version"] == 1
    for 키 in ("python", "numpy", "platform", "seed", "kernel_scope",
               "measured", "measured_build_array", "masks", "budget"):
        assert 키 in loaded, f"{키}가 bench.json에 없다"

    for 키 in ("n_rounds", "mask_mode", "wall_sec", "eps_per_sec", "n_decisions",
               "decisions_per_round", "us_per_decision", "ev_per_round"):
        assert 키 in loaded["measured"]
    assert loaded["measured"]["eps_per_sec"] > 0.0

    for 키 in ("table_view_ns", "build_array_ns", "call_speedup",
               "round_speedup", "values_equal"):
        assert 키 in loaded["masks"]
    assert loaded["masks"]["values_equal"] is True

    budget = loaded["budget"]
    assert budget["planned_eps_per_sec"] == 20_000.0
    assert budget["safety_factor"] == 2.0
    assert len(budget["planned_rows"]) == len(budget["measured_rows"]) == 8
    assert budget["safe_total_hours"] == pytest.approx(
        budget["measured_total_hours"] * 2.0)


def test_마크다운_출력에_세_개의_표가_들어있다():
    result = bench.run_bench(n_rounds=1_000, n_calls=2_000, seed=3)
    md = bench.render_markdown(result)
    assert "## 1. 라운드 시뮬레이션 실측" in md
    assert "## 2. legal_actions 두 방식" in md
    assert "## 3. 계산 예산표" in md
    assert "합계" in md
