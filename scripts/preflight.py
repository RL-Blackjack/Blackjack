"""이 파일은 배포된 서버가 발표에 쓸 수 있는 상태인지 한 번에 점검한다.
입력: 서버 주소(인자) 또는 테스트가 넘긴 httpx 호환 클라이언트.
출력: 항목별 OK/FAIL 한 줄씩과 종료 코드(전부 OK 면 0).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.rules import RULES_V1  # noqa: E402

# 왜 13인가: 강화학습 1 + 지도학습 12. 계획서 ⑥이 모델을 늘리면 이 수를 같이 올린다.
기대모델수: int = 13


@dataclass(frozen=True)
class 결과:
    이름: str
    ok: bool
    설명: str


def 점검(클라) -> list[결과]:
    """항목마다 요청 하나. 예외도 FAIL 한 줄로 만든다."""
    목록: list[결과] = []

    def 더하기(이름: str, 조건: bool, 설명: str) -> None:
        목록.append(결과(이름, bool(조건), 설명))

    try:
        h = 클라.get("/api/health")
        본문 = h.json() if h.status_code == 200 else {}
        더하기("health", h.status_code == 200 and 본문.get("status") == "ok", f"HTTP {h.status_code}")
        더하기("rules_fp", 본문.get("rules_fp") == RULES_V1.fingerprint(),
              f"서버 {본문.get('rules_fp')} / 코드 {RULES_V1.fingerprint()}")
        더하기("models_loaded", int(본문.get("models_loaded", 0)) >= 기대모델수,
              f"{본문.get('models_loaded')}개 (기대 {기대모델수})")
        m = 클라.get("/api/models")
        모델들 = m.json() if m.status_code == 200 else []
        더하기("models", m.status_code == 200 and len(모델들) >= 기대모델수, f"{len(모델들)}개")
        더하기("models_order",
              bool(모델들) and 모델들[0]["exact_ev"] == max(x["exact_ev"] for x in 모델들),
              "첫 항목이 가장 잘 두는 모델(기본 상대)")
        i = 클라.get("/")
        더하기("index", i.status_code == 200
              and i.headers.get("content-type", "").startswith("text/html"),
              f"HTTP {i.status_code}")
        lb = 클라.get("/api/leaderboard")
        더하기("leaderboard", lb.status_code == 200, f"HTTP {lb.status_code}")
    except Exception as e:  # noqa: BLE001
        # 왜 넓게 잡는가: 연결 거부·타임아웃·JSON 오류 어느 것이든 "FAIL 연결" 한 줄이면
        #   발표 30분 전에 알아야 할 것은 다 안 것이다.
        더하기("연결", False, f"{type(e).__name__}: {e}")
    return 목록


def main(argv: list[str]) -> int:
    주소 = argv[1] if len(argv) > 1 else "http://127.0.0.1:8000"
    # 왜 localhost 는 인증서를 안 보는가: 도메인 없이 띄운 Caddy 는 자체 서명 인증서를 쓴다.
    verify = not 주소.startswith("https://localhost") and ".localhost" not in 주소
    with httpx.Client(base_url=주소, timeout=10.0, verify=verify) as 클라:
        목록 = 점검(클라)
    for r in 목록:
        print(f"{'OK  ' if r.ok else 'FAIL'} {r.이름:14s} {r.설명}")
    return 0 if all(r.ok for r in 목록) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
