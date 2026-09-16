"""이 파일은 FastAPI 앱을 조립하고 라우터를 붙인다.
입력: 없음.
출력: create_app()이 만든 FastAPI 인스턴스.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import import_module

from fastapi import FastAPI

from game.api import ROUTER_MODULES
from game.db import create_schema, make_engine


@asynccontextmanager
async def _수명(app: FastAPI) -> AsyncIterator[None]:
    """서버가 뜰 때 표가 없으면 만든다."""
    # 왜 lifespan인가: @app.on_event("startup")은 FastAPI 0.141에서 폐기 예정이다.
    엔진 = make_engine()
    create_schema(엔진)
    엔진.dispose()
    yield


def create_app() -> FastAPI:
    """앱을 만들고 있는 라우터를 전부 붙인다."""
    app = FastAPI(title="블랙잭 강화학습 대전", version="1.0.0", lifespan=_수명)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        # 왜 규칙 지문을 돌려주는가: 서버와 모델이 같은 규칙을 쓰는지 배포 뒤
        #   한 번에 확인할 수 있다. 발표 당일 점검 항목이기도 하다.
        from blackjack_rl.rules import RULES_V1
        return {"status": "ok", "rules_fp": RULES_V1.fingerprint()}

    for 이름 in ROUTER_MODULES:
        전체이름 = f"game.api.{이름}"
        try:
            모듈 = import_module(전체이름)
        except ModuleNotFoundError as e:
            # 왜 이렇게 좁게 잡는가: 아직 안 만든 라우터는 건너뛰되, 라우터
            #   *안에서* 난 import 오류까지 삼키면 안 된다. 그러면 엔드포인트가
            #   조용히 사라져 404가 나고 원인을 못 찾는다.
            if e.name != 전체이름:
                raise
            continue
        app.include_router(모듈.router)

    return app


app = create_app()
