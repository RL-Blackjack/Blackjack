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
    assert read_access_token(make_access_token(42)) == 42


def test_만료된_토큰은_거부된다():
    from game.security import TokenInvalid, make_access_token, read_access_token
    옛날 = datetime.now(timezone.utc) - timedelta(minutes=30)
    with pytest.raises(TokenInvalid):
        read_access_token(make_access_token(42, now=옛날))


def test_유효기간은_15분이다():
    from game.security import make_access_token, read_access_token
    기준 = datetime.now(timezone.utc) - timedelta(minutes=14, seconds=30)
    assert read_access_token(make_access_token(42, now=기준)) == 42


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
