"""이 파일은 argon2 해싱 자리가 다 찼을 때 서버가 기다리지 않고 503으로 물러나는지 확인한다.
입력: 세마포어를 바꿔 끼운 뒤 보낸 로그인 요청.
출력: 503·Retry-After·걸린 시간에 대한 pytest 결과.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import game.security as 보안  # noqa: E402
from test_game_api import 환경  # noqa: E402,F401


def test_해싱_자리가_다_차면_503으로_빨리_거절한다(환경, monkeypatch):
    클라, _ = 환경
    # 왜 자리를 1개로 줄이는가: 실제 4개를 채우려면 스레드 4개가 argon2 안에 멈춰
    #   있어야 하는데, 그 자리에 멈출 방법이 없다. 세마포어 자체를 바꿔 끼운다.
    자리 = threading.BoundedSemaphore(1)
    monkeypatch.setattr(보안, "_해싱자리", 자리)
    monkeypatch.setattr(보안, "ARGON2_WAIT_SECONDS", 0.2)
    assert 자리.acquire(timeout=1)
    try:
        시작 = time.monotonic()
        r = 클라.post("/api/auth/login", json={"email": "a@b.com", "password": "hunter2!!"})
        걸림 = time.monotonic() - 시작
    finally:
        자리.release()
    assert r.status_code == 503, r.text
    assert r.headers.get("retry-after") == "1"
    # 왜 상한이 넉넉한가: 부하가 있는 PC에서 TestClient 왕복이 1초를 넘길 수 있다.
    #   검사하려는 것은 "2초짜리 원래 대기 대신 0.2초 만에 물러났다"는 것뿐이다.
    assert 0.2 <= 걸림 < 3.0
    # 왜: 자리가 나면 바로 정상으로 돌아와야 한다(계정이 없어 401).
    assert 클라.post("/api/auth/login",
                    json={"email": "a@b.com", "password": "hunter2!!"}).status_code == 401


def test_자리는_예외가_나도_돌려준다(monkeypatch):
    # 왜: 검증 중 예외가 자리를 물고 있으면 네 번 뒤에 서버 전체가 503만 낸다.
    # 왜 해셔를 터뜨리는가: 잘못된 해시는 verify_password 안에서 False로 삼켜져 이 경로를
    #   못 찌른다. 아무도 잡지 않는 예외를 일부러 내서 finally 가 자리를 돌려주는지 본다.
    자리 = threading.BoundedSemaphore(1)
    monkeypatch.setattr(보안, "_해싱자리", 자리)

    class 터지는해셔:
        # 왜 객체째 바꾸는가: argon2의 PasswordHasher는 속성 대입을 막는다(setattr 불가).
        def verify(self, *_args, **_kw):
            raise RuntimeError("해셔가 죽었다")

    monkeypatch.setattr(보안, "_해셔", 터지는해셔())
    with pytest.raises(RuntimeError):
        보안.verify_password("$argon2id$whatever", "x")
    assert 자리.acquire(blocking=False)
    자리.release()


def test_기본_자리_수와_대기_시간은_정해진_값이다():
    assert 보안.ARGON2_CONCURRENCY == 4
    assert 보안.ARGON2_WAIT_SECONDS == 2.0
