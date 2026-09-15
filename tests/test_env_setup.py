"""이 파일은 개발 환경의 파이썬·라이브러리 버전이 설계서와 같은지 확인한다.
입력: 현재 실행 중인 파이썬 인터프리터와 설치된 패키지들.
출력: 버전이 다르면 실패하는 pytest 테스트 3개.
"""

import sys

import numpy
import pandas
import pytest


def test_파이썬은_3_14이다():
    # 왜: 3.12와 3.14를 섞어 쓰면 npz 재현 결과가 미묘하게 갈린다. 처음부터 하나로 고정한다.
    assert sys.version_info[0] == 3
    assert sys.version_info[1] == 14


def test_라이브러리_버전이_핀되어_있다():
    assert numpy.__version__ == "2.4.3"
    assert pandas.__version__ == "3.0.1"
    assert pytest.__version__ == "9.0.2"


def test_패키지를_임포트할_수_있다():
    import blackjack_rl

    assert blackjack_rl.__version__ == "0.1.0"
