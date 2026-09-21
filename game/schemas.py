"""이 파일은 API가 주고받는 요청·응답 모양을 정의한다.
입력: 없음(Pydantic 선언만).
출력: 요청·응답 모델들.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator
from pydantic_core import PydanticCustomError

DISPLAY_NAME_MAX: int = 64

# 왜 따로 적는가: 한글 채움 문자들은 범주가 Lo(글자)라 범주 검사로는 걸리지
#   않는데, 화면에는 아무것도 보이지 않아 빈 이름·남의 이름 흉내에 쓰인다.
보이지않는글자: frozenset[str] = frozenset("\u115f\u1160\u3164\uffa0")


class SignupIn(BaseModel):
    """회원가입 요청."""

    email: EmailStr
    # 왜 8자인가: 설계서가 길이를 정하지 않았으므로 여기서 정한다. 너무 짧으면
    #   argon2로 해싱해도 사전 공격에 뚫린다.
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=DISPLAY_NAME_MAX)

    @field_validator("display_name")
    @classmethod
    def _표시명_검사(cls, 값: str) -> str:
        """NFC로 맞추고 앞뒤 공백을 떼어, 보이는 글자만 있는 이름인지 본다."""
        # 왜 검증기에서 떼는가: Field의 길이 검사는 strip보다 먼저 돈다. 핸들러에서
        #   떼면 "   "이 길이 검사를 통과한 뒤 빈 이름('')으로 저장된다(실측 201).
        이름 = unicodedata.normalize("NFC", 값).strip()
        for 글자 in 이름:
            범주 = unicodedata.category(글자)
            # 왜 C*·Zl·Zp인가: 제어(Cc)·서식(Cf, 폭 없는 공백·방향 제어)·서로게이트
            #   ·사용자 정의·미할당 글자와 줄/문단 구분자는 리더보드 화면을 깨거나
            #   보이지 않는 차이로 남의 이름을 흉내 내는 데 쓰인다.
            if 범주[0] == "C" or 범주 in ("Zl", "Zp") or 글자 in 보이지않는글자:
                raise PydanticCustomError(
                    "display_name_invisible", "표시명에 제어 문자나 보이지 않는 문자가 있다")
        if not 이름:
            raise PydanticCustomError("display_name_blank", "표시명이 비어 있다")
        # 왜 다시 재는가: NFC가 글자를 늘릴 수 있다(U+FB2C 한 글자 → 세 글자).
        #   정규화한 값을 저장하므로 64자 칸도 정규화한 값으로 지킨다.
        if len(이름) > DISPLAY_NAME_MAX:
            raise PydanticCustomError(
                "display_name_too_long", "표시명은 {max}자를 넘을 수 없다",
                {"max": DISPLAY_NAME_MAX})
        return 이름


class LoginIn(BaseModel):
    """로그인 요청."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshIn(BaseModel):
    """리프레시·로그아웃 요청."""

    refresh_token: str = Field(min_length=1, max_length=512)


class PasswordChangeIn(BaseModel):
    """비밀번호 변경 요청. 새 비밀번호의 제약은 가입과 같다."""

    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class TokenOut(BaseModel):
    """토큰 발급 응답. 비밀번호나 해시는 절대 담지 않는다."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    """사용자 공개 정보."""

    id: int
    email: str
    display_name: str
    auth_provider: str
    created_at: datetime
