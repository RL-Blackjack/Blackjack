"""이 파일은 게임 생성과 행동 처리, 결정 기록 엔드포인트를 제공한다.
입력: 로그인한 사용자의 행동 요청.
출력: 라운드의 지금 모습과 DP 평가, 그리고 decisions 표의 새 행.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from blackjack_rl.rules import RULES_V1
from blackjack_rl.state import legal_actions
from game.db import get_session
from game.deps import current_user
from game.engine import Finished, IllegalAction, Pending, RoundView, new_seed, play
from game.models import Decision, Game, ModelRegistry, Round, User
from game.optimal import ev_loss, optimal_action
from game.play_schemas import (
    ActIn,
    ActOut,
    Feedback,
    FinishedOut,
    GameOut,
    HandOut,
    NewGameIn,
    OpponentOut,
    PendingOut,
    RoundOut,
)
from game.serving import model_action, store

router = APIRouter(prefix="/api", tags=["games"])


def _저장소(session: Session):
    """모델 저장소가 비어 있으면 DB에서 한 번 채운다."""
    # 왜 여기서 채우는가: lifespan에서 채우면 테스트가 DB를 갈아끼울 때마다
    #   묵은 정책이 남는다. 비어 있을 때만 읽으므로 비용은 처음 한 번뿐이다.
    if not store.serving():
        store.refresh(session)
    return store


def _행동목록(round_: Round) -> list[int]:
    return [int(a) for a in round_.actions.split(",") if a]


def _보기(view: RoundView, seq: int) -> PendingOut | FinishedOut:
    """엔진의 결과를 응답 모양으로 바꾼다."""
    if isinstance(view, Pending):
        return PendingOut(
            seq=seq, total=view.key.total, is_soft=bool(view.key.is_soft),
            dealer_up=view.dealer_up, player_cards=list(view.player_cards),
            legal=list(view.legal), hand_index=view.hand_index,
            n_hands=view.n_hands, bet=view.bet)
    return FinishedOut(
        dealer_up=view.dealer_up, dealer_cards=list(view.dealer_cards),
        net=view.net,
        hands=[HandOut(cards=list(h.cards), total=h.total, bet=h.bet,
                       result=h.result) for h in view.hands])


def _닫기(session: Session, game: Game, round_: Round, view: Finished) -> None:
    """끝난 라운드에 딜러 카드와 순손익을 박고 게임 합계를 갱신한다."""
    round_.is_open = False
    round_.dealer_cards = ",".join(str(c) for c in view.dealer_cards)
    round_.net = float(view.net)
    game.net_result = float(game.net_result) + float(view.net)


def _새라운드(session: Session, game: Game) -> Round:
    """새 라운드를 딜한다. 결정 없이 끝나는 라운드면 바로 닫는다."""
    회차 = 1 + len(list(session.scalars(
        select(Round.id).where(Round.game_id == game.id))))
    시드 = new_seed()
    보기 = play(시드, [])
    라운드 = Round(game_id=game.id, round_no=회차, seed=시드, actions="",
                  is_open=True, dealer_up=int(보기.dealer_up), dealer_cards="",
                  net=0.0)
    session.add(라운드)
    session.flush()
    if isinstance(보기, Finished):
        # 왜: 플레이어나 딜러가 내추럴이면 결정이 하나도 없이 끝난다.
        _닫기(session, game, 라운드, 보기)
    session.commit()
    return 라운드


def _게임찾기(session: Session, game_id: int, user: User) -> Game:
    """내 게임만 찾는다. 남의 것이면 없는 것처럼 군다."""
    게임 = session.get(Game, game_id)
    if 게임 is None or 게임.user_id != user.id:
        # 왜 404인가: 403을 주면 "그 번호의 게임이 존재한다"를 알려 주는 것이고,
        #   번호를 훑어 남이 몇 판 했는지 셀 수 있다.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "그런 게임이 없다")
    return 게임


def _열린라운드(session: Session, game: Game) -> Round | None:
    return session.scalar(
        select(Round).where(Round.game_id == game.id, Round.is_open.is_(True))
        .order_by(Round.round_no.desc()))


def _마지막라운드(session: Session, game: Game) -> Round:
    라운드 = session.scalar(
        select(Round).where(Round.game_id == game.id)
        .order_by(Round.round_no.desc()))
    if 라운드 is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "라운드가 없다")
    return 라운드


def _상대(session: Session, game: Game) -> OpponentOut | None:
    if game.opponent_model_id is None:
        return None
    행 = session.get(ModelRegistry, game.opponent_model_id)
    if 행 is None:
        return None
    return OpponentOut(id=행.id, name=행.name, family=행.family,
                       exact_ev=float(행.exact_ev))


@router.get("/models", response_model=list[OpponentOut])
def list_models(session: Annotated[Session, Depends(get_session)]) -> list[OpponentOut]:
    """서빙 중인 상대 모델을 EV가 좋은 순으로 준다."""
    return [OpponentOut(id=m.id, name=m.name, family=m.family, exact_ev=m.exact_ev)
            for m in _저장소(session).serving()]


@router.post("/games", response_model=GameOut, status_code=status.HTTP_201_CREATED)
def create_game(몸체: NewGameIn,
                사용자: Annotated[User, Depends(current_user)],
                session: Annotated[Session, Depends(get_session)]) -> GameOut:
    """새 게임을 만들고 첫 라운드를 딜한다."""
    저장소 = _저장소(session)
    상대번호 = 몸체.opponent_model_id
    if 상대번호 is None:
        기본 = 저장소.default()
        상대번호 = 기본.id if 기본 is not None else None
    elif 저장소.get(상대번호) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "그런 모델이 없다")

    게임 = Game(user_id=사용자.id, opponent_model_id=상대번호,
               rules_fp=RULES_V1.fingerprint(),
               started_at=datetime.now(timezone.utc), net_result=0.0)
    session.add(게임)
    session.flush()
    라운드 = _새라운드(session, 게임)

    return GameOut(game_id=게임.id, opponent=_상대(session, 게임),
                   round_no=라운드.round_no,
                   round=_보기(play(라운드.seed, _행동목록(라운드)), 0))


@router.get("/games/{game_id}", response_model=GameOut)
def read_game(game_id: int,
              사용자: Annotated[User, Depends(current_user)],
              session: Annotated[Session, Depends(get_session)]) -> GameOut:
    """게임의 지금 모습. 상태는 언제나 시드로 다시 만든다."""
    게임 = _게임찾기(session, game_id, 사용자)
    라운드 = _마지막라운드(session, 게임)
    행동 = _행동목록(라운드)
    return GameOut(game_id=게임.id, opponent=_상대(session, 게임),
                   round_no=라운드.round_no,
                   round=_보기(play(라운드.seed, 행동), len(행동)))


@router.post("/games/{game_id}/rounds", response_model=RoundOut,
             status_code=status.HTTP_201_CREATED)
def deal_round(game_id: int,
               사용자: Annotated[User, Depends(current_user)],
               session: Annotated[Session, Depends(get_session)]) -> RoundOut:
    """같은 게임에서 새 라운드를 딜한다."""
    게임 = _게임찾기(session, game_id, 사용자)
    if _열린라운드(session, 게임) is not None:
        # 왜: 라운드 두 개가 동시에 열려 있으면 어느 쪽에 두는지 알 수 없다.
        raise HTTPException(status.HTTP_409_CONFLICT, "아직 끝나지 않은 라운드가 있다")
    라운드 = _새라운드(session, 게임)
    return RoundOut(round_no=라운드.round_no,
                    round=_보기(play(라운드.seed, []), 0))


@router.post("/games/{game_id}/act", response_model=ActOut)
def act(game_id: int, 몸체: ActIn,
        사용자: Annotated[User, Depends(current_user)],
        session: Annotated[Session, Depends(get_session)]) -> ActOut:
    """행동 하나를 처리하고 DP 정답·EV 손실과 함께 기록한다."""
    게임 = _게임찾기(session, game_id, 사용자)
    라운드 = _열린라운드(session, 게임)
    if 라운드 is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "열려 있는 라운드가 없다")

    행동 = _행동목록(라운드)
    if 몸체.seq != len(행동):
        # 왜: 같은 요청이 두 번 오거나 순서가 뒤엉킨 것이다. 조용히 받아들이면
        #   결정이 두 번 기록돼 일치율이 망가진다.
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"지금 기다리는 결정은 {len(행동)}번인데 {몸체.seq}번이 왔다")

    지금 = play(라운드.seed, 행동)
    if not isinstance(지금, Pending):
        raise HTTPException(status.HTTP_409_CONFLICT, "이 라운드는 이미 끝났다")

    키 = 지금.key
    합법 = legal_actions(키)
    if not 합법[몸체.action]:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            f"지금 고를 수 없는 행동이다: {몸체.action}")

    정답 = optimal_action(키, 합법)
    손실 = ev_loss(키, 몸체.action)
    상대모델 = _저장소(session).get(게임.opponent_model_id or -1)
    ai = model_action(상대모델, 키, 합법) if 상대모델 is not None else None

    session.add(Decision(
        round_id=라운드.id, seq=len(행동),
        total=키.total, is_soft=키.is_soft, dealer_up=키.dealer_up,
        can_double=키.can_double, can_split=키.can_split,
        split_depth=키.split_depth,
        action_taken=int(몸체.action), dp_optimal_action=int(정답),
        dp_ev_loss=float(손실), ai_action=ai,
        created_at=datetime.now(timezone.utc)))

    새행동 = [*행동, int(몸체.action)]
    try:
        다음 = play(라운드.seed, 새행동)
    except IllegalAction as e:
        session.rollback()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    라운드.actions = ",".join(str(a) for a in 새행동)
    if isinstance(다음, Finished):
        _닫기(session, 게임, 라운드, 다음)
    session.commit()

    return ActOut(
        feedback=Feedback(dp_optimal_action=int(정답), dp_ev_loss=float(손실),
                          was_optimal=int(정답) == int(몸체.action), ai_action=ai),
        round=_보기(다음, len(새행동)))


@router.post("/games/{game_id}/finish", status_code=status.HTTP_204_NO_CONTENT)
def finish_game(game_id: int,
                사용자: Annotated[User, Depends(current_user)],
                session: Annotated[Session, Depends(get_session)]) -> Response:
    """게임을 끝낸다. 열린 라운드가 있어도 그대로 닫는다."""
    게임 = _게임찾기(session, game_id, 사용자)
    if 게임.ended_at is None:
        게임.ended_at = datetime.now(timezone.utc)
        session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
