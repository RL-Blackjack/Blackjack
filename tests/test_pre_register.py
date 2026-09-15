"""이 파일은 학습 시작 전 사전등록 JSON이 제대로 만들어지는지 검증한다
입력: scripts/run_dp.py의 build_pre_registration()
출력: pytest 통과/실패
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_dp  # noqa: E402

from blackjack_rl.chartspec import CHART_ROWS, DEALER_COLS  # noqa: E402
from blackjack_rl.reference import TIE_DELTA  # noqa: E402
from blackjack_rl.state import DOUBLE, HIT, SPLIT, STAND  # noqa: E402

허용_사유 = {"deck_model", "rule_diff", "statistical_tie", "undertrained"}


@pytest.fixture(scope="module")
def payload():
    return run_dp.build_pre_registration()


def test_표기를_행동_가치로_바꾸는_규칙이_맞다():
    q = np.array([0.10, 0.20, 0.30, 0.40])
    assert run_dp.value_of_notation(q, "S") == pytest.approx(0.10)
    assert run_dp.value_of_notation(q, "H") == pytest.approx(0.20)
    assert run_dp.value_of_notation(q, "D") == pytest.approx(0.30)
    assert run_dp.value_of_notation(q, "Ds") == pytest.approx(0.30)
    assert run_dp.value_of_notation(q, "Y") == pytest.approx(0.40)
    # 'N'은 스플릿을 뺀 나머지 중 최선이다.
    assert run_dp.value_of_notation(q, "N") == pytest.approx(0.30)


def test_N은_스플릿_칸이_nan이어도_읽힌다():
    q = np.array([0.10, 0.20, np.nan, 0.40])
    assert run_dp.value_of_notation(q, "N") == pytest.approx(0.20)


def test_사유_분류가_TIE_DELTA를_경계로_갈린다():
    assert run_dp.classify(TIE_DELTA / 2) == "statistical_tie"
    assert run_dp.classify(TIE_DELTA * 2) == "deck_model"


def test_사전등록_JSON의_최상위_스키마가_전부_있다(payload):
    for 키 in ("schema_version", "generated_at", "dp_rules_fp", "reference_rules_fp",
               "reference_source", "tie_delta", "n_cells", "n_mismatch", "entries"):
        assert 키 in payload, f"{키}가 사전등록 JSON에 없다"
    assert payload["schema_version"] == 1
    assert payload["n_cells"] == len(CHART_ROWS) * len(DEALER_COLS) == 360
    assert payload["n_mismatch"] == len(payload["entries"])
    assert payload["dp_rules_fp"] != payload["reference_rules_fp"]


def test_각_항목이_여섯_칸을_전부_채운다(payload):
    라벨들 = {row.label for row in CHART_ROWS}
    for entry in payload["entries"]:
        assert set(entry) == {"cell", "dealer", "dp", "ref", "dp_delta", "cause"}
        assert entry["cell"] in 라벨들
        assert entry["dealer"] in DEALER_COLS
        assert entry["dp"] != entry["ref"]
        assert entry["cause"] in 허용_사유
        assert isinstance(entry["dp_delta"], float)


def test_DP_손해는_음수가_아니다(payload):
    # 왜: DP가 고른 행동이 최적이므로 출판표 행동보다 나쁠 수 없다.
    for entry in payload["entries"]:
        assert entry["dp_delta"] >= -1e-9, entry


def test_불일치_칸이_360칸_중_소수다(payload):
    # 왜: 20칸을 넘으면 덱 모형 차이가 아니라 표나 DP에 버그가 있다는 뜻이다.
    assert 0 < payload["n_mismatch"] <= 20


def test_파일로_쓰고_다시_읽어도_같다(tmp_path, payload):
    path = run_dp.write_pre_registration(payload, tmp_path / "expected_mismatch.json")
    assert path.exists()
    다시 = json.loads(path.read_text(encoding="utf-8"))
    assert 다시 == payload


def test_요약표에_모든_불일치_줄이_들어간다(payload):
    md = run_dp.render_summary(payload)
    assert "| 셀 | 딜러 | DP | 출판표 | DP 손해 | 사유 |" in md
    for entry in payload["entries"]:
        assert entry["cell"] in md
    assert md.count("\n") == 5 + len(payload["entries"])


def test_행동_상수가_설계서와_같다():
    assert (STAND, HIT, DOUBLE, SPLIT) == (0, 1, 2, 3)
