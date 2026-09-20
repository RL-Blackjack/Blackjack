"""이 파일은 F4 인증 수정이 나중에 약해지지 않도록 지키는 보강 검사를 모은다.
입력: 로그인 더미 해시와 회원가입 표시명 문자열.
출력: 더미 해시의 온전함과 보이지 않는 문자 거부에 대한 pytest 결과.
"""

import sys
import unicodedata
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


def test_더미_해시는_실제_해셔_설정의_온전한_해시다():
    from game.api.auth import _더미해시
    from game.security import _해셔
    # 왜: 모양만 "$argon2id$"로 시작하는 가짜 문자열이어도 기존 검사(접두사·호출
    #     횟수)는 통과한다. 그러면 없는 계정의 검증이 형식 오류로 곧장 빠져나가
    #     응답이 다시 20배 빨라진다(실측 37.6ms 대 1.7ms). 지금 해셔 설정
    #     그대로 만든 해시여야 비용이 저절로 따라간다.
    assert _해셔.check_needs_rehash(_더미해시) is False


def test_더미_해시를_임의_비밀번호로_검증하면_불일치로_끝난다():
    from argon2.exceptions import VerifyMismatchError

    from game.api.auth import _더미해시
    from game.security import _해셔
    # 왜 InvalidHashError가 아니라 VerifyMismatchError여야 하는가: 잘리거나
    #     흉내만 낸 해시는 형식 검사에서 먼저 죽어 argon2 연산을 건너뛴다.
    #     "비밀번호가 다르다"로 끝나야 해시 계산을 끝까지 한 것이다.
    with pytest.raises(VerifyMismatchError):
        _해셔.verify(_더미해시, "이것은 더미 해시의 비밀번호가 아니다")


# 왜 이 문자들인가: F4 검증자가 남은 구멍으로 지목한 다섯 가지다. 한글 채움
#   문자 U+1160·U+FFA0은 범주가 Lo(글자)라 범주 검사로는 걸리지 않고, 문단
#   구분자 U+2029는 가운데에 끼면 strip이 떼지 못하며, U+E000(사용자 정의)과
#   U+FDD0(영원히 미할당인 비문자)은 화면에 두부 상자나 아무것도 아닌 것으로
#   나와 남의 이름을 그대로 흉내 내는 데 쓰인다.
보이지않는_표시명: list[str] = [
    "\u1160", "a\u1160b", "\uffa0", "a\uffa0b",
    "a\u2029b", "a\ue000b", "a\ufdd0b",
]


def _가입본문(이름: str) -> dict[str, str]:
    """표시명만 바꾼 회원가입 요청 본문을 만든다."""
    return {"email": "jiwoo@example.com", "password": "hunter2!!",
            "display_name": 이름}


def test_보이지_않는_문자가_든_표시명은_검증기에서_막힌다():
    from pydantic import ValidationError

    from game.schemas import SignupIn
    for 이름 in 보이지않는_표시명:
        with pytest.raises(ValidationError) as 잡음:
            SignupIn(**_가입본문(이름))
        # 왜 오류 종류까지 보는가: 길이나 빈 문자열로 우연히 막히면 그 문자를
        #     실제로 거른다는 증거가 못 된다.
        assert 잡음.value.errors()[0]["type"] == "display_name_invisible", repr(이름)


def test_거부한_문자들의_유니코드_범주가_그대로다():
    # 왜: 유니코드 판이 올라 U+FDD0이 배정되거나 U+1160의 범주가 바뀌면 위
    #     검사가 다른 이유로 통과할 수 있다. 거부 근거를 함께 못박는다.
    assert unicodedata.category("\u1160") == "Lo"
    assert unicodedata.category("\uffa0") == "Lo"
    assert unicodedata.category("\u2029") == "Zp"
    assert unicodedata.category("\ue000") == "Co"
    assert unicodedata.category("\ufdd0") == "Cn"


def test_보이는_이름은_넓힌_거부_목록에도_걸리지_않는다():
    from game.schemas import SignupIn
    # 왜: 거부 목록을 넓히면 멀쩡한 이름까지 막기 쉽다. 이모지(So)·라틴 결합
    #     문자·사이 공백이 든 이름은 정규화만 거쳐 그대로 통과해야 한다.
    for 이름 in ["지우", "a b", "🙂 star", "  지우  ", "José", "Ünïcødé"]:
        결과 = SignupIn(**_가입본문(이름)).display_name
        assert 결과 == unicodedata.normalize("NFC", 이름).strip(), repr(이름)
