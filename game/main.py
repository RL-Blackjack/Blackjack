"""이 파일은 FastAPI 앱을 조립하고 라우터를 붙인다.
입력: 없음.
출력: create_app()이 만든 FastAPI 인스턴스.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import import_module
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from game.api import ROUTER_MODULES
from game.db import create_schema, make_engine, make_session_factory
from game.serving import store

log = logging.getLogger("game.main")


def _실을수있게(값: Any) -> Any:
    """응답에 실어도 되게 짝 없는 서로게이트를 바꾼다. 문자열이 아니면 그대로 둔다."""
    # 왜: 짝 없는 서로게이트가 msg나 loc에 하나라도 남으면 starlette가 utf-8로
    #   인코딩하다 죽어, 422로 끝났어야 할 요청이 500이 된다. 지금 쓰는 pydantic
    #   2.13·email-validator는 문제 글자를 코드포인트(U+D800)로 바꿔 적어 실제로는
    #   원문이 섞이지 않는다(실측). 라이브러리가 원문을 되비추는 쪽으로 바뀌어도
    #   500이 나지 않게 두는 방어 코드다.
    if isinstance(값, str):
        return 값.encode("utf-8", "replace").decode("utf-8")
    return 값


@asynccontextmanager
async def _수명(app: FastAPI) -> AsyncIterator[None]:
    """서버가 뜰 때 표가 없으면 만들고 서빙할 모델을 메모리에 올린다."""
    # 왜 lifespan인가: @app.on_event("startup")은 FastAPI 0.141에서 폐기 예정이다.
    엔진 = make_engine()
    create_schema(엔진)
    with make_session_factory(엔진)() as 세션:
        # 왜 기동 때 읽는가: 첫 손님이 파일 13개를 여느라 기다리지 않게 하고, 모델이
        #   하나도 안 올라가는 배포를 첫 요청이 아니라 기동에서 알아채려고. 행이
        #   있는데 전부 실패하면 refresh가 RuntimeError를 내고, 여기서 잡지 않는다.
        올라온수, 실패 = store.refresh(세션)
    엔진.dispose()
    if 올라온수 == 0:
        # 왜 경고만인가: 레지스트리가 비어 있는 것은 아직 등록 전이라는 뜻이다
        #   (scripts/register_models.py). 개발 중 첫 기동까지 막을 이유는 없다.
        log.warning("서빙할 모델이 없다. scripts/register_models.py로 등록해야 한다")
    elif 실패:
        log.warning("모델 %d개를 서빙에서 뺐다: %s", len(실패),
                    ", ".join(f"{f.name}[{f.reason}]" for f in 실패))
    yield


def create_app() -> FastAPI:
    """앱을 만들고 있는 라우터를 전부 붙인다."""
    app = FastAPI(title="블랙잭 강화학습 대전", version="1.0.0", lifespan=_수명)

    @app.exception_handler(RequestValidationError)
    async def _검증오류(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422 본문에 type·loc·msg만 담는다. 거부한 입력값과 ctx는 되돌려주지 않는다."""
        # 왜 input을 빼는가: FastAPI 기본 처리기는 거부한 입력을 그대로 응답에 싣는다.
        #   NaN·Infinity는 allow_nan=False 직렬화에서, 짝 없는 서로게이트는 utf-8
        #   인코딩에서 터져, 익명으로 낼 수 있는 500이 16건 났다(실측). 되비출 이유도
        #   없다 — 보낸 쪽이 이미 아는 값이고, 로그와 응답에 비밀번호가 새는 길이다.
        항목 = [{"type": _실을수있게(오류.get("type")),
                "loc": [_실을수있게(칸) for 칸 in 오류.get("loc", ())],
                "msg": _실을수있게(str(오류.get("msg", "")))}
               for 오류 in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": 항목})

    @app.get("/api/health")
    def health() -> dict[str, str | int]:
        # 왜 규칙 지문을 돌려주는가: 서버와 모델이 같은 규칙을 쓰는지 배포 뒤
        #   한 번에 확인할 수 있다. 발표 당일 점검 항목이기도 하다.
        from blackjack_rl.rules import RULES_V1
        # 왜 모델 수인가: 레지스트리에 행이 있어도 파일이 깨지거나 바꿔치기되면 그
        #   모델만 조용히 빠진다(F6). 이 숫자가 기대보다 작으면 요청 한 번으로 안다.
        return {"status": "ok", "rules_fp": RULES_V1.fingerprint(),
                "models_loaded": len(store.serving())}

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
