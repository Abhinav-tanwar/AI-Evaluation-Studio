FROM node:22-bookworm-slim AS frontend
WORKDIR /web
RUN npm install --global pnpm@11.25.0
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm run build

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --create-home --uid 10001 studio
COPY studio/ ./studio/
COPY data/ ./data/
COPY --from=frontend /web/dist/ ./frontend/dist/
RUN mkdir /app/runtime && chown studio:studio /app/runtime
USER studio
EXPOSE 8040
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8040/api/health')"
CMD ["uvicorn", "studio.app:app", "--host", "0.0.0.0", "--port", "8040"]
