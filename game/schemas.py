"""이 파일은 API가 주고받는 요청·응답 모양을 정의한다.
입력: 없음(Pydantic 선언만).
출력: 요청·응답 모델들.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class SignupIn(BaseModel):
    """회원가입 요청."""

    email: EmailStr
    # 왜 8자인가: 설계서가 길이를 정하지 않았으므로 여기서 정한다. 너무 짧으면
    #   argon2로 해싱해도 사전 공격에 뚫린다.
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=64)


class LoginIn(BaseModel):
    """로그인 요청."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class RefreshIn(BaseModel):
    """리프레시·로그아웃 요청."""

    refresh_token: str = Field(min_length=1, max_length=512)


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
