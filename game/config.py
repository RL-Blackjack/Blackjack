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
    # 왜 빈 문자열이 기본인가: 구글 클라이언트 ID가 없으면 구글 버튼을 아예 안 그린다.
    #   개발 PC에서는 없는 것이 정상이고, 비밀값이 아니라 .env에 그대로 적어도 된다.
    google_client_id: str = ""
    # 왜 env 인가: 배포(prod)에서는 SQLite 와 /docs 를 막는다. dev 가 기본이라 테스트와
    #   개발 PC 는 아무것도 안 바뀐다.
    env: str = "dev"

    @property
    def is_prod(self) -> bool:
        return self.env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """환경변수를 읽어 설정을 만든다. 한 번 만들면 계속 같은 것을 돌려준다."""
    비밀키 = os.environ.get("BJ_JWT_SECRET", "")
    if len(비밀키) < MIN_SECRET_LEN:
        raise ValueError(
            f"BJ_JWT_SECRET이 {MIN_SECRET_LEN}자 이상이어야 한다(현재 {len(비밀키)}자). "
            "예: python -c \"import secrets; print(secrets.token_urlsafe(48))\"")

    환경 = os.environ.get("BJ_ENV", "dev").strip().lower()
    if 환경 not in ("dev", "prod"):
        raise ValueError(f"BJ_ENV 는 dev 또는 prod 여야 한다(현재 {환경!r})")
    주소 = os.environ.get("BJ_DATABASE_URL", DEFAULT_SQLITE_URL)
    if 환경 == "prod" and not 주소.startswith("postgresql+psycopg://"):
        # 왜 막는가: 집 PC에서 SQLite 로 서비스하면 동시 쓰기가 잠기고 백업 절차도
        #   PG 기준이다. 드라이버도 psycopg 3 하나로 고정해 "왜 안 붙지"를 없앤다.
        raise ValueError("BJ_ENV=prod 에서는 BJ_DATABASE_URL 이 "
                         "postgresql+psycopg:// 로 시작해야 한다")

    return Settings(
        database_url=주소,
        jwt_secret=비밀키,
        google_client_id=os.environ.get("BJ_GOOGLE_CLIENT_ID", "").strip(),
        env=환경,
    )
