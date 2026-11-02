import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import BoundedSemaphore
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .evaluation import load_data, run_comparison
from .schemas import BASELINE, CANDIDATE, ReviewRequest, RunRequest
from .settings import Settings
from .storage import Store

ROOT = Path(__file__).resolve().parent.parent


def create_app(settings=None):
    settings = settings or Settings()
    store = Store(settings.database_url)
    lock = BoundedSemaphore(1)
    executor = ThreadPoolExecutor(max_workers=1)

    @asynccontextmanager
    async def lifespan(app):
        for run in store.list():
            if run.get("status") in {"queued", "running"}:
                store.put(
                    {**run, "status": "failed", "error": "Interrupted by service restart. Start a new run."}
                )
        yield
        executor.shutdown(wait=True)
        store.engine.dispose()

    app = FastAPI(title="AI Evaluation Studio", version="1.0.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme not in {"http", "https"} or parsed.netloc != request.headers.get("host"):
                    return JSONResponse(
                        {"detail": "Cross origin API access is not permitted."}, status_code=403
                    )
            remote = request.client and request.client.host not in {"127.0.0.1", "::1", "testclient"}
            if request.url.path != "/api/health" and (remote or settings.studio_api_token):
                supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
                expected = settings.studio_api_token.get_secret_value() if settings.studio_api_token else ""
                if not expected or not secrets.compare_digest(supplied, expected):
                    return JSONResponse({"detail": "A studio access token is required."}, status_code=401)
            if request.method in {"POST", "PUT", "PATCH"}:
                # Count streamed bytes; do not trust a claimed Content-Length.
                body = bytearray()
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > 16384:
                        return JSONResponse({"detail": "Request exceeds 16 KB."}, status_code=413)
                request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def existing(run_id):
        try:
            UUID(run_id)
        except ValueError:
            raise HTTPException(404, "Experiment not found.") from None
        result = store.get(run_id)
        if not result:
            raise HTTPException(404, "Experiment not found.")
        return result

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "live_inference_enabled": bool(settings.allow_live_api and settings.openai_api_key),
            "tracking_enabled": bool(settings.mlflow_tracking_uri),
        }

    @app.get("/api/benchmark")
    def benchmark():
        data, corpus, fingerprint = load_data()
        return {
            **data,
            "corpus": corpus,
            "sha256": fingerprint,
            "defaults": {"baseline": BASELINE.model_dump(), "candidate": CANDIDATE.model_dump()},
        }

    @app.get("/api/runs")
    def runs():
        return [
            {k: v for k, v in r.items() if k != "pipelines"}
            | {"summary": {label: p["metrics"] for label, p in r.get("pipelines", {}).items()}}
            for r in store.list()
        ]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        return {**existing(run_id), "reviews": store.get_reviews(run_id)}

    @app.post("/api/runs", status_code=202)
    def start(request: RunRequest):
        if request.provider == "openai" and not (settings.allow_live_api and settings.openai_api_key):
            raise HTTPException(422, "Live inference is disabled. Configure the backend before using OpenAI.")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "An evaluation is already running. Wait for it to finish.")
        run_id = str(uuid4())
        pending = {
            "id": run_id,
            "name": request.name,
            "status": "queued",
            "created_at": datetime.now(UTC).isoformat(),
            "split": request.split,
            "provider": request.provider,
            "config": request.model_dump(),
        }
        try:
            store.put(pending)

            def work():
                try:
                    store.put({**pending, "status": "running"})
                    result = run_comparison(request, settings)
                    store.put({**result, "id": run_id, "status": "completed"})
                except Exception:  # noqa: BLE001 — isolate background job failures without exposing secrets.
                    store.put(
                        {
                            **pending,
                            "status": "failed",
                            "error": "Evaluation failed. Check the backend configuration and start a new run.",
                        }
                    )
                finally:
                    lock.release()

            executor.submit(work)
        except Exception:
            lock.release()
            raise
        return pending

    @app.get("/api/runs/{run_id}/export")
    def export(run_id: str):
        return Response(
            json.dumps({**existing(run_id), "reviews": store.get_reviews(run_id)}, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="evaluation-{run_id}.json"'},
        )

    @app.post("/api/runs/{run_id}/reviews")
    def review(run_id: str, body: ReviewRequest):
        result = existing(run_id)
        ids = {r["case_id"] for r in result.get("pipelines", {}).get("candidate", {}).get("rows", [])}
        if body.case_id not in ids:
            raise HTTPException(422, "Case is not part of this completed evaluation.")
        store.review(run_id, {**body.model_dump(), "created_at": datetime.now(UTC).isoformat()})
        return {"saved": True}

    @app.post("/api/runs/{run_id}/tracking")
    def tracking(run_id: str):
        result = existing(run_id)
        if result.get("status") != "completed":
            raise HTTPException(409, "Only completed evaluations can be exported.")
        try:
            from .tracking import export_mlflow

            return {"tracking_run_id": export_mlflow(result, settings.mlflow_tracking_uri)}
        except (ImportError, ValueError):
            raise HTTPException(
                422, "Install tracking dependencies and configure MLFLOW_TRACKING_URI."
            ) from None

    dist = ROOT / "frontend/dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(dist / "index.html")

    return app


app = create_app()
