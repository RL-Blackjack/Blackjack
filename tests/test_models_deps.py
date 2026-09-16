"""이 파일은 지도학습에 필요한 라이브러리와 패키지 뼈대가 준비됐는지 확인한다.
입력: 설치된 패키지들.
출력: 버전이 다르거나 패키지가 없으면 실패하는 pytest 결과.
"""

import pytest


def test_라이브러리_버전이_핀되어_있다():
    import joblib
    import sklearn

    assert sklearn.__version__ == "1.9.1"
    assert joblib.__version__ == "1.6.0"


def test_대시보드_라이브러리도_설치되어_있다():
    import plotly
    import streamlit

    assert streamlit.__version__ == "1.64.0"
    assert plotly.__version__ == "7.1.0"


def test_models_패키지를_임포트할_수_있다():
    import blackjack_rl.models as m

    assert m.__doc__ is not None


def test_코어는_여전히_streamlit_없이_돈다():
    # 왜: src/blackjack_rl/ 안에서 streamlit을 import하면 pytest가 무거워지고
    #     계획서 ④의 게임 서버도 쓸데없이 streamlit을 끌고 오게 된다.
    import os
    import subprocess
    import sys
    from pathlib import Path

    # 왜 PYTHONPATH를 직접 넘기는가: pyproject의 pythonpath=["src"]는 pytest 자신의
    #   sys.path에만 적용되고 환경변수에는 반영되지 않는다. 이 저장소는 패키지를
    #   설치하지 않으므로, 새 인터프리터에 경로를 넘기지 않으면 import 자체가 실패해
    #   "streamlit 없이 도는가"를 검사하기도 전에 죽는다.
    src = str(Path(__file__).resolve().parents[1] / "src")
    환경 = dict(os.environ, PYTHONPATH=src, PYTHONIOENCODING="utf-8")

    코드 = (
        "import sys; "
        "sys.modules['streamlit'] = None; "
        "sys.modules['plotly'] = None; "
        "import blackjack_rl, blackjack_rl.models; "
        "print('ok')"
    )
    결과 = subprocess.run([sys.executable, "-c", 코드],
                         capture_output=True, text=True, env=환경)
    assert 결과.returncode == 0, 결과.stderr
    assert "ok" in 결과.stdout
