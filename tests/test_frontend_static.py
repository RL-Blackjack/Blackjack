"""이 파일은 정적 프런트가 서빙되고, JS가 부르는 API 경로가 실제로 존재하는지 확인한다.
입력: TestClient 로 받은 정적 파일과 app.routes.
출력: 200 응답·경로 일치·innerHTML 미사용에 대한 pytest 결과.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from test_game_api import 환경  # noqa: E402,F401

정적 = ROOT / "game" / "static"
JS들 = sorted(정적.glob("*.js"))
경로정규식 = r'["`](/api/[^"`]*)["`]'


def test_첫_화면과_자산이_서빙된다(환경):
    클라, _ = 환경
    r = 클라.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>" in r.text
    # Task 5에서는 둘뿐이다. Task 6이 app.js·table.js, Task 7이 pages.js를 더한다.
    for 이름 in ["style.css", "logic.js"]:
        assert 클라.get(f"/{이름}").status_code == 200, 이름


def test_API_경로는_정적_파일에_가려지지_않는다(환경):
    클라, _ = 환경
    assert 클라.get("/api/health").status_code == 200
    assert 클라.get("/api/없는것").status_code == 404


def test_JS가_부르는_API_경로가_전부_존재한다(환경):
    클라, _ = 환경
    실제 = {r.path for r in 클라.app.routes if getattr(r, "path", "").startswith("/api")}
    # 왜: 화면이 부르는 경로가 서버에서 이름이 바뀌면 브라우저에서만 404가 난다.
    #   문자열 리터럴을 긁어 라우트와 대조하면 그 어긋남을 테스트가 잡는다.
    # 왜 [^"`]* 인가: 템플릿 리터럴 안의 ${게임.game_id} 처럼 한글이 섞이거나
    #   ?limit=1 같은 질의가 붙은 경로도 잡아야 한다. 좁은 문자 집합은 조용히 건너뛴다.
    for js in JS들:
        for 경로 in re.findall(경로정규식, js.read_text("utf-8")):
            정규 = re.sub(r"\$\{[^}]+\}", "{game_id}", 경로).split("?")[0]
            assert 정규 in 실제, f"{js.name}: {경로}"
    # 왜 파일마다 개수를 세는가: 정규식이 아무것도 못 잡으면 위 for 문은 조용히 통과한다.
    #   table.js는 게임 경로 4개, app.js는 인증·로비 6개 이상, pages.js는 5개 이상을 부른다.
    하한 = {"table.js": 4, "app.js": 6, "pages.js": 5}
    for 이름, 최소 in 하한.items():
        if (정적 / 이름).exists():
            잡힌 = re.findall(경로정규식, (정적 / 이름).read_text("utf-8"))
            assert len(잡힌) >= 최소, (이름, 잡힌)


def test_innerHTML을_쓰지_않는다():
    # 왜: 표시명은 남이 지은 문자열이다. innerHTML 한 줄이면 리더보드가 XSS 통로가 된다.
    for 파일 in JS들 + [정적 / "index.html"]:
        assert "innerHTML" not in 파일.read_text("utf-8"), 파일.name


def test_외부_스크립트는_구글_하나뿐이다():
    본문 = (정적 / "index.html").read_text("utf-8")
    외부 = re.findall(r'<script[^>]+src="(https?://[^"]+)"', 본문)
    assert 외부 == ["https://accounts.google.com/gsi/client"]
