"""이 파일은 모델 서빙이 정책 배열만 읽고 불법 행동을 내지 않는지 확인한다.
입력: models/ 의 npz 파일과 메모리 DB.
출력: 로딩·선택·레지스트리 동기화에 대한 pytest 결과.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from blackjack_rl.rules import RULES_V1  # noqa: E402
from blackjack_rl.state import REACHABLE_KEYS, legal_actions  # noqa: E402
from game.db import make_engine  # noqa: E402
from game.models import Base, ModelRegistry  # noqa: E402
from game.serving import KEY_INDEX, ModelStore, ServedModel, model_action  # noqa: E402

지금 = datetime.now(timezone.utc)


@pytest.fixture
def 세션():
    e = make_engine("sqlite://")      # 외래키가 켜진 서버와 같은 엔진
    Base.metadata.create_all(e)
    with Session(e) as s:
        yield s


def 더미정책(값: int = 0) -> np.ndarray:
    return np.full(len(REACHABLE_KEYS), 값, dtype=np.int8)


def test_키_색인이_610칸이고_순서가_REACHABLE와_같다():
    assert len(KEY_INDEX) == 610
    for i, k in enumerate(REACHABLE_KEYS):
        assert KEY_INDEX[k] == i


def test_정책이_고른_행동을_그대로_돌려준다():
    정책 = 더미정책(0)
    키 = REACHABLE_KEYS[0]
    정책[KEY_INDEX[키]] = 1      # HIT
    모델 = ServedModel(id=1, name="t", family="rl", exact_ev=-0.01, policy=정책)
    assert model_action(모델, 키, legal_actions(키)) == 1


def test_불법_행동을_고르면_합법으로_되돌아간다():
    # 왜: 인간 모방 모델(계획서 ⑥)은 사람의 버릇을 배우므로 스플릿이 안 되는
    #     자리에서 스플릿을 고를 수 있다. 그때 서버가 죽으면 안 된다.
    키 = next(k for k in REACHABLE_KEYS if not legal_actions(k)[3])
    정책 = 더미정책(3)          # 전부 SPLIT — 이 키에서는 불법이다
    모델 = ServedModel(id=1, name="t", family="imitation", exact_ev=-0.1, policy=정책)
    고른것 = model_action(모델, 키, legal_actions(키))
    assert legal_actions(키)[고른것]


def test_색인에_없는_상태는_스탠드로_되돌아간다():
    from blackjack_rl.state import StateKey
    # 왜 하드 3인가: 카드 두 장의 최소 합계가 4라 절대 나올 수 없는 상태다(실측:
    #     REACHABLE_KEYS의 최소 합계 4). 하드 21은 도달 가능한 키가 10개 있어 쓸 수 없다.
    없는키 = StateKey(total=3, is_soft=0, dealer_up=2,
                     can_double=1, can_split=0, split_depth=0)
    assert 없는키 not in KEY_INDEX
    모델 = ServedModel(id=1, name="t", family="rl", exact_ev=-0.01,
                      policy=더미정책(1))
    # 왜 STAND인가: 모르는 자리에서 모델의 판단을 믿을 근거가 없다. STAND는
    #     언제나 합법이고 카드를 더 받지 않으므로 가장 덜 해롭다.
    assert model_action(모델, 없는키, legal_actions(없는키)) == 0


def test_저장된_모델_열세_개가_전부_불법_행동을_내지_않는다():
    from blackjack_rl.models.registry import load_policy
    파일들 = sorted((ROOT / "models").glob("*.npz"))
    assert len(파일들) >= 12, "계획서 ③의 지도학습 모델이 없다"
    합법표 = np.stack([legal_actions(k) for k in REACHABLE_KEYS])
    for f in 파일들:
        정책, _카드 = load_policy(f, rules=RULES_V1)
        불법 = int(np.sum(~합법표[np.arange(len(REACHABLE_KEYS)), 정책]))
        assert 불법 == 0, f"{f.name}이 불법 행동을 {불법}칸에서 낸다"


def test_저장소가_서빙_중인_모델만_읽는다(세션, tmp_path):
    from blackjack_rl.models.registry import ModelCard, save_policy

    for 이름, 서빙 in [("보임", True), ("숨김", False)]:
        경로 = tmp_path / f"{이름}.npz"
        save_policy(경로, 더미정책(0), ModelCard(
            name=이름, family="rl", rules_fp=RULES_V1.fingerprint(),
            exact_ev=-0.01, policy_sha256="", n_train_labels=0,
            test_acc=0.0, fit_seconds=0.0, n_params=0,
            created_at=지금.isoformat()))
        세션.add(ModelRegistry(name=이름, family="rl",
                              rules_fp=RULES_V1.fingerprint(),
                              artifact_path=str(경로), artifact_sha256="0" * 64,
                              exact_ev=-0.01, is_serving=서빙))
    세션.commit()

    저장소 = ModelStore()
    저장소.refresh(세션)
    assert [m.name for m in 저장소.serving()] == ["보임"]


def test_규칙이_다른_모델은_거부된다(세션, tmp_path):
    from blackjack_rl.models.registry import ModelCard, ModelRulesMismatch, save_policy
    경로 = tmp_path / "옛규칙.npz"
    save_policy(경로, 더미정책(0), ModelCard(
        name="옛규칙", family="rl", rules_fp="ffffffffffff", exact_ev=-0.01,
        policy_sha256="", n_train_labels=0, test_acc=0.0, fit_seconds=0.0,
        n_params=0, created_at=지금.isoformat()))
    세션.add(ModelRegistry(name="옛규칙", family="rl", rules_fp="ffffffffffff",
                          artifact_path=str(경로), artifact_sha256="0" * 64,
                          exact_ev=-0.01, is_serving=True))
    세션.commit()

    # 왜 예외인가: 규칙이 다른 모델을 태연히 서빙하면 DP 정답과 EV 손실이
    #     전부 틀린 값으로 쌓인다. 조용히 넘어가는 것이 가장 나쁘다.
    with pytest.raises(ModelRulesMismatch):
        ModelStore().refresh(세션)


def test_기본_상대는_EV가_가장_좋은_모델이다(세션, tmp_path):
    from blackjack_rl.models.registry import ModelCard, save_policy
    for 이름, ev in [("약함", -0.09), ("강함", -0.005)]:
        경로 = tmp_path / f"{이름}.npz"
        save_policy(경로, 더미정책(0), ModelCard(
            name=이름, family="rl", rules_fp=RULES_V1.fingerprint(), exact_ev=ev,
            policy_sha256="", n_train_labels=0, test_acc=0.0, fit_seconds=0.0,
            n_params=0, created_at=지금.isoformat()))
        세션.add(ModelRegistry(name=이름, family="rl",
                              rules_fp=RULES_V1.fingerprint(),
                              artifact_path=str(경로), artifact_sha256="0" * 64,
                              exact_ev=ev, is_serving=True))
    세션.commit()
    저장소 = ModelStore()
    저장소.refresh(세션)
    assert 저장소.default().name == "강함"


def test_게임_서버는_sklearn을_부르지_않는다():
    import subprocess
    코드 = ("import sys; import game.serving; "
           "sys.exit(1 if 'sklearn' in sys.modules else 0)")
    # 왜 PYTHONPATH를 넘기는가: pyproject의 pythonpath는 pytest에만 적용되므로
    #     맨손 subprocess는 src를 못 찾는다(계획서 ③에서 겪은 그 문제다).
    결과 = subprocess.run([sys.executable, "-c", 코드], cwd=str(ROOT),
                        env={"PYTHONPATH": f"{ROOT}{';' if sys.platform == 'win32' else ':'}{ROOT / 'src'}",
                             "SYSTEMROOT": "C:\\Windows", "PATH": ""},
                        capture_output=True, text=True)
    assert 결과.returncode == 0, f"serving이 sklearn을 끌고 온다: {결과.stderr}"


def test_강화학습_스냅샷을_레지스트리_형식으로_바꾼다(tmp_path):
    from scripts.register_models import convert_rl_snapshot
    from blackjack_rl.models.registry import load_policy

    경로 = convert_rl_snapshot(ROOT / "artifacts", tmp_path)
    정책, 카드 = load_policy(경로, rules=RULES_V1)
    assert 정책.shape == (610,)
    assert 카드.family == "rl"
    assert 카드.n_train_labels == 0     # 강화학습은 레이블을 한 장도 안 봤다
    # 왜 이 값인가: 300만 에피소드 스냅샷의 마지막 프레임을 DP 정책평가로 잰
    #     정확 EV다(시뮬레이션 아님, 분산 0). 계획서 ②에서 확정된 수치다.
    assert 카드.exact_ev == pytest.approx(-0.008822, abs=5e-6)


def test_에피소드가_가장_많은_스냅샷을_고른다(tmp_path):
    from scripts.register_models import convert_rl_snapshot
    from blackjack_rl.models.registry import load_policy
    # 왜: 계획서 ③에서 sorted(glob)[-1]이 200k짜리 시험 산출물을 집어 기준선이
    #     4.6배 틀어진 적이 있다. 이름 정렬이 아니라 에피소드 최댓값으로 골라야 한다.
    _정책, 카드 = load_policy(convert_rl_snapshot(ROOT / "artifacts", tmp_path),
                            rules=RULES_V1)
    assert 카드.exact_ev > -0.02, "200k 스냅샷을 잘못 골랐을 때 나오는 값이다"


def test_레지스트리_동기화가_두_번_돌아도_중복이_안_생긴다(세션):
    from scripts.register_models import sync_registry
    첫번째 = sync_registry(세션, ROOT / "models")
    두번째 = sync_registry(세션, ROOT / "models")
    assert 첫번째 >= 12
    assert 첫번째 == 두번째
    assert len(list(세션.scalars(select(ModelRegistry)))) == 첫번째
