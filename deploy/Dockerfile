# deploy/Dockerfile — api 와 research 가 같은 이미지를 쓴다(같은 코어, 같은 의존성).
FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONPATH=/app/src:/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY game/ game/
COPY app/ app/
COPY scripts/ scripts/
COPY models/ models/
COPY artifacts/ artifacts/
COPY reports/ reports/
# 왜 root 가 아닌가: 컨테이너가 뚫려도 파일 시스템을 못 건드린다.
RUN useradd --create-home bj && chown -R bj /app
USER bj
EXPOSE 8000 8501
