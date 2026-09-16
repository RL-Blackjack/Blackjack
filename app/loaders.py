"""이 파일은 대시보드가 읽을 산출물을 찾아 불러온다.
입력: artifacts/ 의 npz 와 reports/ 의 비교 보고서.
출력: 학습 스냅샷, 모델 비교 보고서, DP 해.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
COMPARISON_PATH = REPORTS_DIR / "supervised_comparison.json"

SCHEMA_VERSION = 1


def list_runs(directory: Path | str = ARTIFACTS_DIR) -> list[Path]:
    """학습 결과 npz 목록을 이름순으로 돌려준다."""
    폴더 = Path(directory)
    if not 폴더.exists():
        return []
    return sorted(폴더.glob("*.npz"))


def read_comparison(path: Path | str = COMPARISON_PATH) -> dict | None:
    """모델 비교 보고서를 읽는다. 없으면 None, 포맷이 다르면 예외."""
    경로 = Path(path)
    if not 경로.exists():
        return None
    실린값 = json.loads(경로.read_text(encoding="utf-8"))
    if 실린값.get("schema_version") != SCHEMA_VERSION:
        # 왜 예외인가: 포맷이 바뀐 옛 보고서를 조용히 그리면 화면의 숫자가
        #   틀리게 된다. 발표 중에 그런 일이 나면 알아차릴 방법이 없다.
        raise ValueError(
            f"보고서 스키마가 {실린값.get('schema_version')}인데 "
            f"{SCHEMA_VERSION}을 기대한다. scripts/train_supervised.py를 다시 돌려라.")
    return 실린값


# ── 아래부터는 streamlit이 있을 때만 쓰는 캐시 래퍼다 ──
# 왜 이렇게 나누는가: 위의 순수 함수들은 streamlit 없이 테스트할 수 있어야 한다.

def _캐시_데이터(func):
    try:
        import streamlit as st
        return st.cache_data(func)
    except Exception:
        return func


def _캐시_자원(func):
    try:
        import streamlit as st
        return st.cache_resource(func)
    except Exception:
        return func


@_캐시_데이터
def load_snapshot(path: str):
    """학습 스냅샷 npz 하나를 읽는다."""
    from blackjack_rl.train.snapshot import load_run
    return load_run(Path(path))


@_캐시_데이터
def load_comparison() -> dict | None:
    """모델 비교 보고서를 읽는다."""
    return read_comparison()


@_캐시_자원
def load_dp():
    """DP 해를 한 번만 푼다. 수 초 걸리지만 결과가 불변이다."""
    from blackjack_rl.dp.exact import solve_optimal
    from blackjack_rl.rules import RULES_V1
    return solve_optimal(RULES_V1)
