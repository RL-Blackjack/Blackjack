"""이 파일은 모델 서빙이 레지스트리와 어긋난 파일을 빼고 레지스트리 변경을 따라가는지 확인한다.
입력: tmp 폴더에 만든 npz 정책 파일과 메모리 DB.
출력: 해시·이름 대조, 지문 기반 다시 읽기, 실패 유형별 격리에 대한 pytest 결과.
"""

import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import game.serving as 서빙  # noqa: E402
from blackjack_rl.models.registry import ModelCard, load_policy, save_policy  # noqa: E402
from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.state import REACHABLE_KEYS  # noqa: E402
from game.db import make_engine  # noqa: E402
from game.models import Base, ModelRegistry  # noqa: E402
from game.serving import ModelArtifactMismatch, ModelStore, file_sha256  # noqa: E402

지금 = datetime.now(timezone.utc)


@pytest.fixture
def 세션():
    e = make_engine("sqlite://")      # 외래키가 켜진 서버와 같은 엔진
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s
    e.dispose()


def 정책파일(경로: Path, 카드이름: str, 값: int = 0) -> Path:
    """한 값으로 채운 정책을 지금 규칙의 카드와 함께 저장한다."""
    return save_policy(경로, np.full(len(REACHABLE_KEYS), 값, dtype=np.int8), ModelCard(
        name=카드이름, family="rl", rules_fp=RULES_V1.fingerprint(), exact_ev=-0.01,
        policy_sha256="", n_train_labels=0, test_acc=0.0, fit_seconds=0.0,
        n_params=0, created_at=지금.isoformat()))


def 등록(세션, 폴더: Path, 이름: str, *, 카드이름: str | None = None) -> ModelRegistry:
    """정책 파일을 만들고 그 순간의 파일 해시로 레지스트리에 올린다."""
    경로 = 정책파일(폴더 / f"{이름}.npz", 카드이름 or 이름)
    행 = ModelRegistry(name=이름, family="rl", rules_fp=RULES_V1.fingerprint(),
                      artifact_path=경로.name, artifact_sha256=file_sha256(경로),
                      exact_ev=-0.01, is_serving=True)
    세션.add(행)
    세션.commit()
    return 행


def ERROR가_있다(caplog, 조각: str) -> bool:
    return any(r.levelno == logging.ERROR and 조각 in r.getMessage()
               for r in caplog.records)


def 세며읽기(monkeypatch) -> list[int]:
    """game.serving의 load_policy 호출을 센다. 목록 길이가 호출 수다."""
    읽은횟수: list[int] = []
    원래 = 서빙.load_policy

    def 감싼것(*a, **k):
        읽은횟수.append(1)
        return 원래(*a, **k)

    monkeypatch.setattr(서빙, "load_policy", 감싼것)
    return 읽은횟수


def test_파일을_바꿔치기하면_그_모델은_서빙에서_빠진다(세션, tmp_path, caplog):
    멀쩡 = 등록(세션, tmp_path, "멀쩡")
    바뀜 = 등록(세션, tmp_path, "바뀜")
    # 왜 같은 이름·같은 규칙으로 다시 저장하는가: load_policy의 규칙·내용 해시 검사와
    #     카드 이름 대조를 모두 통과하는 파일이다. 레지스트리 해시 대조만 잡을 수 있다.
    정책파일(tmp_path / "바뀜.npz", "바뀜", 값=1)
    assert load_policy(tmp_path / "바뀜.npz", rules=RULES_V1)[1].name == "바뀜"

    저장소 = ModelStore(tmp_path)
    with caplog.at_level(logging.ERROR, logger="game.serving"):
        올라온수, 실패 = 저장소.refresh(세션)
    assert 올라온수 == 1
    assert [m.name for m in 저장소.serving()] == ["멀쩡"]
    assert 저장소.get(바뀜.id) is None
    assert [(f.name, f.reason) for f in 실패] == [("바뀜", "sha_mismatch")]
    assert isinstance(실패[0].error, ModelArtifactMismatch)
    assert ERROR가_있다(caplog, "바뀜")
    # 왜 해시를 싣는가: 게임 행에 박아 ai_action이 어느 파일에서 나왔는지 가린다.
    assert 저장소.get(멀쩡.id).artifact_sha256 == file_sha256(tmp_path / "멀쩡.npz")


def test_다른_모델의_파일로_덮어써도_빠진다(세션, tmp_path, caplog):
    # 왜: 누락 탐색에서 mlp_random_20.npz를 rl_mc.npz로 복사하자 rl_mc 자리에서
    #     다른 모델이 그대로 서빙됐다. 파일 자체는 멀쩡해서 load_policy가 통과한다.
    등록(세션, tmp_path, "갑")
    등록(세션, tmp_path, "을")
    (tmp_path / "을.npz").write_bytes((tmp_path / "갑.npz").read_bytes())
    저장소 = ModelStore(tmp_path)
    with caplog.at_level(logging.ERROR, logger="game.serving"):
        _올라온수, 실패 = 저장소.refresh(세션)
    assert [m.name for m in 저장소.serving()] == ["갑"]
    assert [(f.name, f.reason) for f in 실패] == [("을", "sha_mismatch")]
    assert ERROR가_있다(caplog, "을")


def test_카드_이름이_레지스트리와_다르면_빠진다(세션, tmp_path, caplog):
    등록(세션, tmp_path, "멀쩡")
    # 왜 해시는 맞추는가: 파일을 잘못 짝지어 등록한 경우다. 해시는 그 파일 그대로라
    #     해시 대조로는 못 잡고, 카드가 스스로 밝힌 이름으로만 잡힌다.
    등록(세션, tmp_path, "갑", 카드이름="을")
    저장소 = ModelStore(tmp_path)
    with caplog.at_level(logging.ERROR, logger="game.serving"):
        올라온수, 실패 = 저장소.refresh(세션)
    assert 올라온수 == 1
    assert [m.name for m in 저장소.serving()] == ["멀쩡"]
    assert [(f.name, f.reason) for f in 실패] == [("갑", "name_mismatch")]
    assert isinstance(실패[0].error, ModelArtifactMismatch)
    assert ERROR가_있다(caplog, "갑")


def _없애기(경로: Path) -> None:
    경로.unlink()


def _반토막(경로: Path) -> None:
    경로.write_bytes(경로.read_bytes()[: 경로.stat().st_size // 2])


@pytest.mark.parametrize("망가뜨리기, 해시를_새로_박나, 사유, 예외", [
    (_없애기, False, "missing", FileNotFoundError),
    (_반토막, False, "sha_mismatch", ModelArtifactMismatch),     # 등록한 뒤 잘렸다
    (_반토막, True, "load_error", zipfile.BadZipFile),           # 잘린 채로 등록됐다
], ids=["missing", "truncated_after_register", "registered_truncated"])
def test_망가진_파일은_유형과_무관하게_그_모델만_빠진다(세션, tmp_path, caplog,
                                           망가뜨리기, 해시를_새로_박나, 사유, 예외):
    등록(세션, tmp_path, "멀쩡")
    깨짐 = 등록(세션, tmp_path, "깨짐")
    망가뜨리기(tmp_path / "깨짐.npz")
    if 해시를_새로_박나:
        # 왜: 해시 대조를 지나 load_policy까지 가는 손상도 그 모델만 빠져야 한다.
        깨짐.artifact_sha256 = file_sha256(tmp_path / "깨짐.npz")
        세션.commit()

    저장소 = ModelStore(tmp_path)
    with caplog.at_level(logging.ERROR, logger="game.serving"):
        올라온수, 실패 = 저장소.refresh(세션)
    assert 올라온수 == 1
    assert [m.name for m in 저장소.serving()] == ["멀쩡"]
    assert [(f.name, f.reason, f.path.name) for f in 실패] == [("깨짐", 사유, "깨짐.npz")]
    assert isinstance(실패[0].error, 예외)
    assert ERROR가_있다(caplog, "깨짐")


def test_레지스트리가_바뀌면_다시_읽는다(세션, tmp_path):
    갑 = 등록(세션, tmp_path, "갑")
    을 = 등록(세션, tmp_path, "을")
    저장소 = ModelStore(tmp_path)
    assert 저장소.ensure_fresh(세션) is 저장소
    assert sorted(m.name for m in 저장소.serving()) == ["갑", "을"]

    을.is_serving = False                 # 운영자가 서빙을 끈다
    세션.commit()
    저장소.ensure_fresh(세션)
    assert [m.name for m in 저장소.serving()] == ["갑"]
    assert 저장소.get(을.id) is None

    # 왜 해시만 바꾸는가: 재학습해 같은 이름으로 덮어쓰면 행의 id·서빙 여부는 그대로고
    #     해시만 바뀐다. 누락 탐색에서는 이때 옛 정책이 계속 서빙됐다.
    정책파일(tmp_path / "갑.npz", "갑", 값=1)
    갑.artifact_sha256 = file_sha256(tmp_path / "갑.npz")
    세션.commit()
    저장소.ensure_fresh(세션)
    assert 저장소.get(갑.id).artifact_sha256 == 갑.artifact_sha256
    assert np.all(저장소.get(갑.id).policy == 1)

    을.is_serving = True                  # 다시 켜면 돌아온다
    세션.commit()
    저장소.ensure_fresh(세션)
    assert sorted(m.name for m in 저장소.serving()) == ["갑", "을"]


def test_레지스트리가_그대로면_다시_읽지_않는다(세션, tmp_path, monkeypatch):
    등록(세션, tmp_path, "갑")
    을 = 등록(세션, tmp_path, "을")
    읽은횟수 = 세며읽기(monkeypatch)
    저장소 = ModelStore(tmp_path)
    저장소.ensure_fresh(세션)
    assert len(읽은횟수) == 2             # 처음에는 두 모델을 읽는다

    읽은횟수.clear()
    for _ in range(5):                    # 요청마다 게임 API가 부르는 것과 같다
        저장소.ensure_fresh(세션)
    assert 읽은횟수 == []
    assert len(저장소.serving()) == 2

    을.is_serving = False                 # 세는 장치가 살아 있는지도 확인한다
    세션.commit()
    저장소.ensure_fresh(세션)
    assert len(읽은횟수) == 1


def test_지문은_id_해시_서빙여부를_id_순으로_담는다(세션, tmp_path):
    갑 = 등록(세션, tmp_path, "갑")
    을 = 등록(세션, tmp_path, "을")
    을.is_serving = False
    세션.commit()
    assert ModelStore(tmp_path).registry_fingerprint(세션) == (
        (갑.id, 갑.artifact_sha256, True), (을.id, 을.artifact_sha256, False))


def test_reset하면_지문이_같아도_다시_읽는다(세션, tmp_path, monkeypatch):
    갑 = 등록(세션, tmp_path, "갑")
    저장소 = ModelStore(tmp_path)
    저장소.ensure_fresh(세션)
    읽은횟수 = 세며읽기(monkeypatch)

    # 왜: store는 모듈 전역이라 pytest 한 프로세스에서 DB를 갈아끼워도 남는다.
    #     픽스처가 시작 때 비우면 앞 테스트의 모델이 새지 않는다.
    저장소.reset()
    assert (저장소.loaded, 저장소.serving(), 저장소.get(갑.id)) == (False, [], None)
    저장소.ensure_fresh(세션)
    assert [m.name for m in 저장소.serving()] == ["갑"]
    assert len(읽은횟수) == 1


def test_전부_실패해도_요청마다_다시_읽지_않는다(세션, tmp_path, monkeypatch):
    깨짐 = 등록(세션, tmp_path, "깨짐")
    (tmp_path / "깨짐.npz").write_bytes(b"\x00" * 100)
    깨짐.artifact_sha256 = file_sha256(tmp_path / "깨짐.npz")     # load_policy까지 가게
    세션.commit()
    읽은횟수 = 세며읽기(monkeypatch)
    저장소 = ModelStore(tmp_path)

    # 왜: 행이 있는데 하나도 못 올리면 첫 시도는 크게 실패해야 하지만, 그 뒤 요청마다
    #     파일을 다시 열고 ERROR를 쏟으면 안 된다(F6의 loaded와 같은 약속).
    with pytest.raises(RuntimeError, match="깨짐"):
        저장소.ensure_fresh(세션)
    for _ in range(3):
        저장소.ensure_fresh(세션)
    assert len(읽은횟수) == 1

    정책파일(tmp_path / "깨짐.npz", "깨짐")      # 다시 등록하면 지문이 바뀌어 살아난다
    깨짐.artifact_sha256 = file_sha256(tmp_path / "깨짐.npz")
    세션.commit()
    저장소.ensure_fresh(세션)
    assert [m.name for m in 저장소.serving()] == ["깨짐"]


def test_등록_스크립트는_서버의_해시_함수를_쓴다():
    import scripts.register_models as 등록스크립트
    # 왜: 등록 때와 서빙 때 해시 계산이 한 함수여야 둘이 어긋나지 않는다. 게임 서버가
    #     scripts/를 import하지 않도록 함수는 game/serving.py에 둔다.
    assert 등록스크립트.file_sha256 is 서빙.file_sha256
