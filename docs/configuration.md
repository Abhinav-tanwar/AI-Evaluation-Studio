# Configuration

Settings load from a private root `.env` file or environment variables. `.env` is ignored by Git and Docker builds. Never put provider keys in the frontend.

## Optional live OpenAI inference

1. Add a private `OPENAI_API_KEY` to `.env`.
2. Set `ALLOW_LIVE_API=true` and choose an available `OPENAI_MODEL`.
3. Restart the backend. The OpenAI adapter becomes selectable in New comparison.
4. Set `INPUT_USD_PER_MILLION` and `OUTPUT_USD_PER_MILLION` to the applicable model rates if a cost estimate is needed. Without both rates, costs remain unknown.

A development comparison makes 60 logical requests, a holdout comparison 40; the SDK can retry each request once. These calls are billable and send fictional passages and questions to OpenAI. The adapter uses [Responses structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) with `store=False`. Live inference was not exercised in the published verification; the request contract and usage accounting are covered with a test double. Model availability and pricing can change.

## MLflow tracking

```sh
pip install -r requirements-tracking.txt
```

Set `MLFLOW_TRACKING_URI` to an existing tracking server and restart. Select **Log to MLflow** after a completed comparison. For a local file store, use an absolute `file:///...` URI and set `MLFLOW_ALLOW_FILE_STORE=true` if required by your MLflow version. Run `mlflow ui --backend-store-uri file:///absolute/path` to inspect the local store. Tracking records include pipeline metrics and `evaluation.json`. No external tracking server is contacted unless configured.

## Docker with PostgreSQL

Set a strong `POSTGRES_PASSWORD` and `STUDIO_API_TOKEN` in `.env`, then:

```sh
docker compose up --build -d
```

Open `http://127.0.0.1:8041` and use Connection settings to enter the studio token. This token is distinct from an OpenAI key and stays in browser memory. The container disables live API calls by default. Stop with `docker compose down`; named volumes retain history. Do not delete volumes unless you intend to delete stored experiments.

## Boundaries

This application is for a trusted local operator. It binds to loopback in the documented setup. Requests from nonlocal clients require a configured token; protected API calls reject cross origin browser requests. Host validation, 16 KB request limits, one active evaluation, and sanitized provider errors reduce accidental exposure. Docker runs as a nonroot user, drops capabilities, and exposes no database port.

These controls are not multitenant isolation. There is no account system, TLS termination, distributed job runner, or per user authorization. Do not expose it directly to the public internet. Review dependencies and configure a proper authenticated gateway before any shared deployment.
