"""이 파일은 DP 정답표를 레이블로 삼아 지도학습 모델 두 개를 학습시킨다.
입력: 특징 행렬 X, 레이블 y, 훈련/시험 분할.
출력: 학습된 모델과 정확도·학습시간을 담은 TrainedModel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from blackjack_rl.models.splits import Split

MODEL_KINDS: tuple[str, str] = ("rf", "mlp")

RF_TREES: int = 200
MLP_HIDDEN: tuple[int, int] = (64, 64)
MLP_MAX_ITER: int = 2000


@dataclass
class TrainedModel:
    """학습이 끝난 모델 하나와 그 성적표."""

    kind: str
    estimator: object
    split: Split
    seed: int
    train_acc: float
    test_acc: float
    fit_seconds: float
    n_train_labels: int

    def policy_full(self, X: np.ndarray) -> np.ndarray:
        """610칸 전체에 대한 행동 예측을 정책 배열로 돌려준다."""
        return self.estimator.predict(X).astype(np.int8)

    def n_params(self) -> int:
        """모델이 들고 있는 파라미터(또는 노드) 수. 비용 비교용."""
        if self.kind == "rf":
            # 왜 노드 수인가: 트리 모델에는 가중치가 없다. 저장·추론 비용을
            #   가장 잘 대변하는 것이 전체 노드 수다.
            return int(sum(t.tree_.node_count for t in self.estimator.estimators_))
        return int(sum(w.size for w in self.estimator.coefs_)
                   + sum(b.size for b in self.estimator.intercepts_))

    def name(self) -> str:
        return f"{self.kind}_{self.split.name()}"


def make_estimator(kind: str, seed: int):
    """종류 이름으로 sklearn 추정기를 만든다."""
    if kind == "rf":
        return RandomForestClassifier(n_estimators=RF_TREES, random_state=seed, n_jobs=1)
    if kind == "mlp":
        # 왜 max_iter가 2000인가: 기본값 200으로는 수렴 경고가 뜨고 정확도가 낮다.
        #   610개짜리 작은 데이터라 2000까지 돌려도 1초 미만이다.
        return MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                             random_state=seed)
    raise ValueError(f"kind는 {MODEL_KINDS} 중 하나여야 한다: {kind!r}")


def train(kind: str, X: np.ndarray, y: np.ndarray, split: Split, seed: int) -> TrainedModel:
    """분할대로 학습하고 훈련·시험 정확도를 잰다."""
    est = make_estimator(kind, seed)

    t0 = time.perf_counter()
    est.fit(X[split.train_idx], y[split.train_idx])
    걸린시간 = time.perf_counter() - t0

    return TrainedModel(
        kind=kind, estimator=est, split=split, seed=seed,
        train_acc=float(est.score(X[split.train_idx], y[split.train_idx])),
        test_acc=float(est.score(X[split.test_idx], y[split.test_idx])),
        fit_seconds=걸린시간, n_train_labels=len(split.train_idx),
    )
