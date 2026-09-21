"""이 파일은 구글 ID 토큰을 검증해 사용자 신원을 꺼낸다.
입력: 브라우저의 Google Identity Services 버튼이 준 ID 토큰(JWT)과 우리 클라이언트 ID.
출력: GoogleIdentity 또는 TokenInvalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import jwt

from game.security import TokenInvalid

GOOGLE_JWKS_URL: str = "https://www.googleapis.com/oauth2/v3/certs"
# 왜 둘인가: 구글 문서가 두 발급자 문자열을 모두 유효하다고 적는다.
GOOGLE_ISSUERS: tuple[str, str] = ("https://accounts.google.com", "accounts.google.com")

_키집_캐시: jwt.PyJWKClient | None = None


def _키집() -> jwt.PyJWKClient:
    """구글 공개키 묶음. 처음 한 번 받고 캐시한다(테스트는 이 함수를 바꿔 끼운다)."""
    global _키집_캐시
    if _키집_캐시 is None:
        _키집_캐시 = jwt.PyJWKClient(GOOGLE_JWKS_URL, cache_keys=True)
    return _키집_캐시


@dataclass(frozen=True)
class GoogleIdentity:
    """검증을 통과한 구글 계정의 최소 정보."""

    sub: str
    email: str
    email_verified: bool
    name: str


def verify_google_id_token(token: str, *, client_id: str) -> GoogleIdentity:
    """서명·발급자·대상·만료를 검증하고 신원을 돌려준다. 어떤 실패든 TokenInvalid."""
    try:
        키 = _키집().get_signing_key_from_jwt(token)
        # 왜 세 가지를 다 요구하는가: aud가 없으면 남의 앱에 발급된 토큰이 통과하고,
        #   iss가 없으면 구글이 아닌 서명자를 못 거르고, exp가 없으면 영원히 산다.
        본문 = jwt.decode(token, 키.key, algorithms=["RS256"], audience=client_id,
                        issuer=list(GOOGLE_ISSUERS),
                        options={"require": ["exp", "iat", "sub", "aud", "iss", "email"]})
        return GoogleIdentity(sub=str(본문["sub"]), email=str(본문["email"]),
                              email_verified=bool(본문.get("email_verified", False)),
                              name=str(본문.get("name", "")))
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as e:
        # 왜 사유를 합치는가: "만료"와 "대상 불일치"를 구분해 주면 위조 시도에 힌트가 된다.
        raise TokenInvalid("구글 토큰을 쓸 수 없다") from e
