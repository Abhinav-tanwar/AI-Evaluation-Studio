import hashlib
import json
import math
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

from .engine import Retriever, openai_predict, reference_predict
from .schemas import RunRequest

ROOT = Path(__file__).resolve().parent.parent


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def load_data():
    benchmark_bytes = (ROOT / "data/benchmark.json").read_bytes()
    corpus_bytes = (ROOT / "data/corpus.json").read_bytes()
    return (
        json.loads(benchmark_bytes),
        json.loads(corpus_bytes),
        hashlib.sha256(benchmark_bytes + corpus_bytes).hexdigest(),
    )


def score(case, output, retrieved, corpus):
    expected = set(case["expected_sources"])
    actual = set(output.citations)
    available = {p["id"] for p in retrieved}
    source_text = {p["id"]: p["text"] for p in corpus}
    valid = bool(actual) and actual <= available
    # Deliberately strict: copied evidence text is supported; paraphrases require human review.
    supported = (
        bool(normalize(output.answer))
        and valid
        and any(normalize(output.answer) in normalize(source_text.get(i, "")) for i in actual)
    )
    correct = normalize(output.answer) == normalize(case["expected_answer"]) and output.abstained == (
        not expected
    )
    return {
        "exact_match": int(correct),
        "retrieval_recall": len(expected & available) / len(expected) if expected else None,
        "citation_valid": int(valid) if not output.abstained else None,
        "verbatim_support": int(supported) if not output.abstained else None,
        "abstention_correct": int(output.abstained) if not expected else None,
        "invalid_citations": sorted(actual - available),
    }


def aggregate(rows):
    def avg(key):
        values = [r["scores"][key] for r in rows if r["scores"].get(key) is not None]
        return mean(values) if values else None

    latencies = sorted(r["latency_ms"] for r in rows)
    usages = [r["usage"] for r in rows]
    costs = [r["estimated_cost_usd"] for r in rows]
    return {
        "cases": len(rows),
        "passed": sum(r["scores"]["exact_match"] for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        **{
            k: avg(k)
            for k in [
                "exact_match",
                "retrieval_recall",
                "citation_valid",
                "verbatim_support",
                "abstention_correct",
            ]
        },
        "median_latency_ms": median(latencies),
        "p95_latency_ms": latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)],
        "input_tokens": sum(u["input_tokens"] for u in usages)
        if all(u["input_tokens"] is not None for u in usages)
        else None,
        "output_tokens": sum(u["output_tokens"] for u in usages)
        if all(u["output_tokens"] is not None for u in usages)
        else None,
        "estimated_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
    }


def release_gate(baseline, candidate, tolerance=0.0, latency_budget_ms=5000):
    checks = [
        {
            "name": "No exact match regression",
            "passed": candidate["exact_match"] >= baseline["exact_match"] - tolerance,
        },
        {
            "name": "No retrieval recall regression",
            "passed": (candidate["retrieval_recall"] or 0) >= (baseline["retrieval_recall"] or 0),
        },
        {"name": "No provider errors", "passed": candidate["errors"] == 0},
        {"name": "p95 within 5,000 ms budget", "passed": candidate["p95_latency_ms"] <= latency_budget_ms},
    ]
    return {
        "passed": all(x["passed"] for x in checks),
        "checks": checks,
        "exact_match_tolerance": tolerance,
        "latency_budget_ms": latency_budget_ms,
    }


def run_comparison(request: RunRequest, settings=None, client=None):
    benchmark, corpus, fingerprint = load_data()
    cases = [c for c in benchmark["cases"] if c["split"] == request.split]
    retriever = Retriever(corpus)
    if request.provider == "openai":
        if settings is None or not settings.allow_live_api or not settings.openai_api_key:
            raise ValueError("Live inference requires ALLOW_LIVE_API=true and a backend OPENAI_API_KEY.")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), timeout=30, max_retries=1)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unversioned", True
    result = {
        "id": str(uuid4()),
        "name": request.name,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": benchmark["version"],
        "dataset_sha256": fingerprint,
        "split": request.split,
        "provider": request.provider,
        "revision": revision,
        "working_tree_dirty": dirty,
        "model": settings.openai_model if request.provider == "openai" else "extractive-v1",
        "pricing": {
            "input_usd_per_million": settings.input_usd_per_million,
            "output_usd_per_million": settings.output_usd_per_million,
        }
        if request.provider == "openai"
        else None,
        "implementation_sha256": hashlib.sha256(
            b"".join(f.name.encode() + f.read_bytes() for f in sorted((ROOT / "studio").glob("*.py")))
        ).hexdigest(),
        "scorer_version": "strict-v1",
        "config": request.model_dump(),
        "pipelines": {},
    }
    for label, config in [("baseline", request.baseline), ("candidate", request.candidate)]:
        rows = []
        for case in cases:
            started = time.perf_counter()
            context = retriever.search(case["question"], config)
            error = None
            usage = {"input_tokens": None, "output_tokens": None}
            output = None
            try:
                if request.provider == "openai":
                    output, usage = openai_predict(case["question"], context, client, settings.openai_model)
                else:
                    output, usage = reference_predict(context)
                scores = score(case, output, context, corpus)
            except Exception as exc:  # noqa: BLE001 — record provider failure without persisting SDK messages.
                # Do not persist SDK exception strings: they may contain request data or secrets.
                error = type(exc).__name__
                scores = {
                    "exact_match": 0,
                    "retrieval_recall": len(set(case["expected_sources"]) & {p["id"] for p in context})
                    / len(case["expected_sources"])
                    if case["expected_sources"]
                    else None,
                    "citation_valid": 0,
                    "verbatim_support": 0,
                    "abstention_correct": 0 if not case["expected_sources"] else None,
                    "invalid_citations": [],
                }
            cost = None
            if request.provider == "reference":
                cost = 0.0
            elif (
                settings.input_usd_per_million is not None
                and settings.output_usd_per_million is not None
                and all(v is not None for v in usage.values())
            ):
                cost = (
                    usage["input_tokens"] * settings.input_usd_per_million
                    + usage["output_tokens"] * settings.output_usd_per_million
                ) / 1_000_000
            rows.append(
                {
                    "case_id": case["id"],
                    "question": case["question"],
                    "category": case["category"],
                    "expected_answer": case["expected_answer"],
                    "expected_sources": case["expected_sources"],
                    "output": output.model_dump() if output else None,
                    "retrieved": context,
                    "scores": scores,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 4),
                    "usage": usage,
                    "estimated_cost_usd": cost,
                    "error": error,
                }
            )
        result["pipelines"][label] = {"name": config.name, "metrics": aggregate(rows), "rows": rows}
    result["gate"] = release_gate(
        *(result["pipelines"][label]["metrics"] for label in ["baseline", "candidate"])
    )
    return result
