"""이 파일은 사전 점검 스크립트가 정상 서버에서 전부 OK, 고장 난 서버에서 FAIL 을 내는지 확인한다.
입력: TestClient 로 감싼 앱과 응답을 바꿔치기한 가짜 클라이언트.
출력: 점검 결과 목록과 종료 코드에 대한 pytest 결과.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.preflight import 기대모델수, 점검  # noqa: E402
from test_game_api import 환경  # noqa: E402,F401


def test_정상_서버는_전부_OK(환경):
    클라, _ = 환경
    결과 = 점검(클라)
    assert all(r.ok for r in 결과), [r for r in 결과 if not r.ok]
    assert {r.이름 for r in 결과} == {"health", "rules_fp", "models_loaded", "models",
                                    "models_order", "index", "leaderboard"}


def test_기대_모델_수는_models_폴더와_같다():
    # 왜: 모델을 늘리고 이 수를 안 올리면 점검이 "13개면 OK"라고 거짓 안심을 준다.
    assert 기대모델수 == len(list((ROOT / "models").glob("*.npz")))


def test_모델이_비면_FAIL(환경, monkeypatch):
    from game.serving import store
    클라, _ = 환경
    monkeypatch.setattr(store, "serving", lambda: [])
    결과 = {r.이름: r.ok for r in 점검(클라)}
    assert 결과["models_loaded"] is False and 결과["health"] is True


def test_연결이_안_되면_FAIL_한_줄이다():
    class 죽은클라:
        def get(self, _path):
            raise ConnectionError("서버가 없다")

    결과 = 점검(죽은클라())
    assert len(결과) == 1 and 결과[0].이름 == "연결" and not 결과[0].ok
