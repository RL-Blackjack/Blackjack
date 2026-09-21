"""이 파일은 배포 파일이 설계서 §8 의 보안 규칙을 지키는지 확인한다.
입력: deploy/ 의 compose·Caddyfile·.env.example 텍스트.
출력: DB 포트 비공개·prod 강제·HSTS·자리표시자에 대한 pytest 결과.
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
배포 = ROOT / "deploy"
compose = yaml.safe_load((배포 / "docker-compose.yml").read_text("utf-8"))
서비스 = compose["services"]


def test_서비스_다섯_개():
    assert set(서비스) == {"db", "api", "research", "caddy", "backup"}


def test_DB와_앱은_포트를_밖에_열지_않는다():
    # 왜: 설계서 §8.1 — Caddy 만 밖을 본다. api 포트가 닫혀 있어야 XFF 를 붙일 수 있는
    #   것이 Caddy 뿐이라 --forwarded-allow-ips='*' 가 안전하다(I3).
    for 이름 in ("db", "api", "research", "backup"):
        assert "ports" not in 서비스[이름], 이름
    assert sorted(서비스["caddy"]["ports"]) == ["443:443", "80:80"]


def test_api는_prod로_뜨고_프록시_헤더를_Caddy에서만_믿는다():
    env = 서비스["api"]["environment"]
    assert env["BJ_ENV"] == "prod"
    assert env["BJ_DATABASE_URL"].startswith("postgresql+psycopg://")
    명령 = " ".join(서비스["api"]["command"])
    assert "--proxy-headers" in 명령 and "--forwarded-allow-ips=*" in 명령
    assert "--workers 1" in 명령   # 왜: 요청 제한기가 프로세스 메모리에 있다
    assert 서비스["api"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_앱은_슈퍼유저가_아니라_전용_계정으로_붙는다():
    # 왜(설계서 §8.2): 공식 이미지의 POSTGRES_USER 는 슈퍼유저다. 앱이 그 계정이면
    #   SQL 주입 한 번에 DB 전체가 넘어간다.
    url = 서비스["api"]["environment"]["BJ_DATABASE_URL"]
    assert url.startswith("postgresql+psycopg://bj_app:")
    assert "./initdb:/docker-entrypoint-initdb.d:ro" in 서비스["db"]["volumes"]
    초기화 = (배포 / "initdb" / "01-app-role.sh").read_text("utf-8")
    assert "NOSUPERUSER" in 초기화 and "bj_app" in 초기화


def test_백업은_매일_돌고_7일_보관한다():
    본문 = (배포 / "backup.sh").read_text("utf-8")
    assert "pg_dump" in 본문 and "-mtime +7" in 본문 and "sleep 86400" in 본문


def test_backup_sh는_POSIX_sh가_받는_변수_이름만_쓰고_LF다():
    # 왜: dash(postgres:17 의 /bin/sh)·bash 는 변수 이름을 [A-Za-z_][A-Za-z0-9_]* 로만 받는다.
    #   `파일=...` 은 문법상 유효한 '명령어'라 `sh -n` 은 0 으로 통과하고 실행 때만 127 로 죽는다.
    #   그래서 정규식으로 대입문 이름을 본다. CRLF: 윈도 호스트에서 바인드 마운트되는 파일이라
    #   CRLF 면 dash 가 `set -eu\r` 에서 죽는다. read_text() 는 개행을 정규화하므로 바이트로 본다.
    for 경로 in (배포 / "backup.sh", 배포 / "initdb" / "01-app-role.sh"):
        assert b"\r" not in 경로.read_bytes(), 경로.name
        코드 = "\n".join(줄 for 줄 in 경로.read_text("utf-8").splitlines()
                       if not 줄.lstrip().startswith("#"))
        이름들 = re.findall(r"^\s*([^\s=]+)=", 코드, re.M)
        for 이름 in 이름들:
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", 이름), (경로.name, 이름)
    assert re.findall(r"^\s*([^\s=]+)=", (배포 / "backup.sh").read_text("utf-8"), re.M), \
        "변수를 하나도 못 찾았다 — 정규식을 확인하라"


def test_gitattributes가_셸_스크립트를_LF로_고정한다():
    # 왜: 이 저장소는 core.autocrlf=true 라 .gitattributes 가 없으면 체크아웃 때 CRLF 가 된다.
    본문 = (ROOT / ".gitattributes").read_text("utf-8")
    assert "*.sh text eol=lf" in 본문 and "deploy/Caddyfile text eol=lf" in 본문


def test_Caddyfile은_두_호스트와_HSTS를_가진다():
    본문 = (배포 / "Caddyfile").read_text("utf-8")
    assert "{$PLAY_HOST}" in 본문 and "{$RESEARCH_HOST}" in 본문
    assert "reverse_proxy api:8000" in 본문 and "reverse_proxy research:8501" in 본문
    assert "Strict-Transport-Security" in 본문
    assert b"\r" not in (배포 / "Caddyfile").read_bytes()


def test_dockerignore가_비밀과_가상환경을_뺀다():
    본문 = (ROOT / ".dockerignore").read_text("utf-8").splitlines()
    for 항목 in (".venv/", "*.db", "deploy/.env", "deploy/backups/", ".git/"):
        assert 항목 in 본문, 항목


def test_env_example에는_진짜_비밀이_없다():
    본문 = (배포 / ".env.example").read_text("utf-8")
    for 키 in ("BJ_JWT_SECRET", "POSTGRES_PASSWORD", "APP_DB_PASSWORD", "PLAY_HOST",
              "RESEARCH_HOST", "BJ_GOOGLE_CLIENT_ID"):
        assert re.search(rf"^{키}=", 본문, re.M), 키
    for 키 in ("BJ_JWT_SECRET", "POSTGRES_PASSWORD", "APP_DB_PASSWORD"):
        값 = re.search(rf"^{키}=(.*)$", 본문, re.M).group(1)
        assert 값.startswith("CHANGE-ME-") and len(값) < 32, 키
    assert ".env\n" in (ROOT / ".gitignore").read_text("utf-8")
    assert "deploy/backups/" in (ROOT / ".gitignore").read_text("utf-8")
