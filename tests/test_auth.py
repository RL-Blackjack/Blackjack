"""이 파일은 비밀번호 해싱·JWT·요청 제한이 설계서 §5.2대로 도는지 확인한다.
입력: 평문 비밀번호와 토큰 문자열.
출력: 해싱·검증·만료·제한에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _설정(monkeypatch):
    monkeypatch.setenv("BJ_JWT_SECRET", "test-secret-at-least-32-characters-long")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_같은_비밀번호도_해시가_다르다():
    from game.security import hash_password
    # 왜: argon2는 해시마다 무작위 소금을 쓴다. 같으면 소금이 없다는 뜻이고,
    #     그러면 무지개표 한 번으로 전체 계정이 뚫린다.
    assert hash_password("hunter2") != hash_password("hunter2")


def test_해시에_평문이_들어있지_않다():
    from game.security import hash_password
    assert "hunter2" not in hash_password("hunter2")


def test_argon2id를_쓴다():
    from game.security import hash_password
    assert hash_password("hunter2").startswith("$argon2id$")


def test_맞는_비밀번호는_통과하고_틀린_것은_막힌다():
    from game.security import hash_password, verify_password
    h = hash_password("hunter2")
    assert verify_password(h, "hunter2") is True
    assert verify_password(h, "hunter3") is False


def test_망가진_해시는_False를_돌려준다():
    from game.security import verify_password
    # 왜: 예외가 밖으로 나가면 500이 되고, 틀린 비밀번호(401)와 구분되는 그
    #     차이만으로 계정이 있는지 없는지가 샌다.
    assert verify_password("이건해시가아니다", "hunter2") is False


def test_이메일은_소문자로_정규화된다():
    from game.security import normalize_email
    # 왜: citext를 쓰지 않기로 했으므로 정규화를 코드가 책임진다.
    assert normalize_email("  JIWOO@Example.COM ") == "jiwoo@example.com"


def test_액세스_토큰을_만들고_되읽는다():
    from game.security import make_access_token, read_access_token
    # 왜 튜플인가: 토큰에 버전(ver)이 실리면서 (사용자 번호, 버전)을 돌려준다.
    assert read_access_token(make_access_token(42)) == (42, 0)
    assert read_access_token(make_access_token(42, 3)) == (42, 3)


def test_만료된_토큰은_거부된다():
    from game.security import TokenInvalid, make_access_token, read_access_token
    옛날 = datetime.now(timezone.utc) - timedelta(minutes=30)
    with pytest.raises(TokenInvalid):
        read_access_token(make_access_token(42, now=옛날))


def test_유효기간은_15분이다():
    from game.security import make_access_token, read_access_token
    기준 = datetime.now(timezone.utc) - timedelta(minutes=14, seconds=30)
    assert read_access_token(make_access_token(42, now=기준)) == (42, 0)


def test_서명이_다르면_거부된다(monkeypatch):
    from game.security import TokenInvalid, make_access_token, read_access_token
    토큰 = make_access_token(42)
    monkeypatch.setenv("BJ_JWT_SECRET", "완전히-다른-비밀키-이것도-32자를-넘게-길게-쓴다")
    get_settings.cache_clear()
    with pytest.raises(TokenInvalid):
        read_access_token(토큰)


def test_쓰레기_토큰은_거부된다():
    from game.security import TokenInvalid, read_access_token
    for 나쁜값 in ["", "a.b.c", "not-a-token", "."]:
        with pytest.raises(TokenInvalid):
            read_access_token(나쁜값)


def test_리프레시_토큰은_원문과_해시가_다르다():
    from game.security import hash_refresh_token, new_refresh_token
    원문, 해시 = new_refresh_token()
    assert 원문 != 해시
    assert len(원문) >= 32
    assert hash_refresh_token(원문) == 해시


def test_리프레시_토큰은_매번_다르다():
    from game.security import new_refresh_token
    assert len({new_refresh_token()[0] for _ in range(50)}) == 50


def test_요청_제한은_분당_10회다():
    from game.security import RateLimiter
    제한 = RateLimiter(limit=10, window_seconds=60.0)
    assert all(제한.allow("1.2.3.4", now=1000.0) for _ in range(10))
    assert 제한.allow("1.2.3.4", now=1000.0) is False


def test_제한은_IP마다_따로_센다():
    from game.security import RateLimiter
    제한 = RateLimiter(limit=2, window_seconds=60.0)
    assert 제한.allow("1.1.1.1", now=0.0) and 제한.allow("1.1.1.1", now=0.0)
    assert 제한.allow("1.1.1.1", now=0.0) is False
    assert 제한.allow("2.2.2.2", now=0.0) is True


def test_창이_지나면_다시_허용된다():
    from game.security import RateLimiter
    제한 = RateLimiter(limit=2, window_seconds=60.0)
    제한.allow("1.1.1.1", now=0.0)
    제한.allow("1.1.1.1", now=0.0)
    assert 제한.allow("1.1.1.1", now=30.0) is False
    assert 제한.allow("1.1.1.1", now=61.0) is True


def test_필수_클레임이_없는_토큰은_거부된다():
    import jwt

    from game.security import ALGORITHM, TokenInvalid, read_access_token
    지금 = int(datetime.now(timezone.utc).timestamp())
    온전 = {"sub": "42", "iat": 지금, "exp": 지금 + 600}
    비밀 = get_settings().jwt_secret
    # 왜 ver 없이도 통과하는가: 서버만 서명하므로 ver가 없는 토큰은 버전 0으로 읽는다.
    assert read_access_token(jwt.encode(온전, 비밀, algorithm=ALGORITHM)) == (42, 0)
    # 왜: exp가 없으면 영원히 유효한 토큰이 된다. 서버만 서명하지만 키가 한 번
    #     새면 되돌릴 수 없으므로 세 클레임을 모두 요구한다.
    for 뺄것 in ["exp", "sub", "iat"]:
        빠짐 = {k: v for k, v in 온전.items() if k != 뺄것}
        with pytest.raises(TokenInvalid):
            read_access_token(jwt.encode(빠짐, 비밀, algorithm=ALGORITHM))


def test_제한기는_빈_키를_지운다():
    from game.security import RateLimiter
    제한 = RateLimiter(limit=2, window_seconds=60.0)
    for 번호 in range(100):
        제한.allow(f"10.0.0.{번호}", now=0.0)
    assert 제한.key_count() == 100
    # 왜: IP를 바꿔 가며 한 번씩만 보내면 키가 영원히 남아 메모리가 계속 는다.
    #     창이 지나 비어 버린 키는 다른 키의 요청 때 함께 지워져야 한다.
    assert 제한.allow("10.0.1.1", now=61.0) is True
    assert 제한.key_count() == 1


def test_제한기_청소는_창_안_기록이_남은_키를_지우지_않는다():
    from game.security import RateLimiter
    제한 = RateLimiter(limit=2, window_seconds=60.0)
    assert 제한.allow("1.1.1.1", now=0.0) is True
    assert 제한.allow("1.1.1.1", now=20.0) is True
    # 왜: 61초에 청소가 돌지만 마지막 기록(20초)이 아직 창 안이다. 키의 첫
    #     기록만 보고 지우면 20초 기록까지 함께 사라져, 20~62초 한 창에 세
    #     번(20·61·62)이 허용된다. 한도가 조용히 늘어나는 회귀다.
    assert 제한.allow("1.1.1.1", now=61.0) is True
    assert 제한.allow("1.1.1.1", now=62.0) is False
    assert 제한.key_count() == 1


def test_제한기는_여러_스레드에서_한도를_넘지_않는다(monkeypatch):
    import threading
    import time

    from game import security
    from game.security import RateLimiter

    def 양보하는_len(값):
        길이 = len(값)
        time.sleep(0)
        return 길이

    # 왜 모듈의 len을 바꿔치는가: 확인(len)과 기록(append) 사이에서 스레드가
    #     바뀌어야 경합이 드러난다. GIL 아래에서는 그 틈이 좁아 잠금이 없어도
    #     통과해 버린다(실측 200회 중 0회 초과). 여기서 양보시키면 잠금 없는
    #     구현은 매번 114~119회를 허용한다.
    monkeypatch.setattr(security, "len", 양보하는_len, raising=False)
    제한 = RateLimiter(limit=100, window_seconds=600.0)
    출발 = threading.Barrier(20)
    허용 = [0] * 20

    def 일꾼(번호: int) -> None:
        출발.wait()
        허용[번호] = sum(제한.allow("1.2.3.4") for _ in range(10))

    스레드들 = [threading.Thread(target=일꾼, args=(i,)) for i in range(20)]
    for t in 스레드들:
        t.start()
    for t in 스레드들:
        t.join()
    assert sum(허용) == 100
