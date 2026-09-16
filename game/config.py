"""이 파일은 게임 서버 설정을 환경변수에서 읽어 한 곳에 모은다.
입력: BJ_DATABASE_URL, BJ_JWT_SECRET 등 환경변수.
출력: 불변 Settings 객체 하나.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SQLITE_URL: str = f"sqlite:///{ROOT / 'game.db'}"

# 왜 32자인가: JWT 서명 키가 짧으면 무차별 대입으로 위조할 수 있다.
#   집 PC를 인터넷에 여는 이상 "secret" 같은 값으로 배포하는 사고를 막아야 한다.
MIN_SECRET_LEN: int = 32


@dataclass(frozen=True)
class Settings:
    """서버 한 벌의 설정. 환경변수에서 읽어 한 번만 만든다."""

    database_url: str
    jwt_secret: str
    access_token_minutes: int = 15
    refresh_token_days: int = 14
    login_rate_per_minute: int = 10
    models_dir: Path = ROOT / "models"
    artifacts_dir: Path = ROOT / "artifacts"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """환경변수를 읽어 설정을 만든다. 한 번 만들면 계속 같은 것을 돌려준다."""
    비밀키 = os.environ.get("BJ_JWT_SECRET", "")
    if len(비밀키) < MIN_SECRET_LEN:
        raise ValueError(
            f"BJ_JWT_SECRET이 {MIN_SECRET_LEN}자 이상이어야 한다(현재 {len(비밀키)}자). "
            "예: python -c \"import secrets; print(secrets.token_urlsafe(48))\"")

    return Settings(
        database_url=os.environ.get("BJ_DATABASE_URL", DEFAULT_SQLITE_URL),
        jwt_secret=비밀키,
    )
