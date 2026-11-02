from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from studio.engine import Retriever, openai_predict, reference_predict
from studio.evaluation import load_data, release_gate, run_comparison, score
from studio.schemas import BASELINE, Prediction, RunRequest
from studio.settings import Settings


def test_dataset_integrity():
    data, corpus, fingerprint = load_data()
    ids = {p["id"] for p in corpus}
    assert len(data["cases"]) == 50 and len(corpus) == 40
    assert len({c["id"] for c in data["cases"]}) == 50
    assert sum(c["split"] == "holdout" for c in data["cases"]) == 20
    assert len(fingerprint) == 64
    for case in data["cases"]:
        assert set(case["expected_sources"]) <= ids
        if case["expected_sources"]:
            assert any(
                p["text"] == case["expected_answer"] for p in corpus if p["id"] in case["expected_sources"]
            )


def test_empty_query_abstains():
    _, corpus, _ = load_data()
    assert Retriever(corpus).search("the and or", BASELINE) == []
    prediction, usage = reference_predict([])
    assert prediction.abstained and not prediction.citations
    assert usage["input_tokens"] is None


@pytest.mark.parametrize(
    "answer,citations,supported",
    [("Example.", ["p1"], 1), ("", ["p1"], 0), ("Invented", ["p1"], 0), ("Example.", ["missing"], 0)],
)
def test_support_does_not_accept_empty_or_invented_evidence(answer, citations, supported):
    passage = {"id": "p1", "text": "Example."}
    case = {"expected_sources": ["p1"], "expected_answer": "Example."}
    result = score(
        case, Prediction(answer=answer, citations=citations, abstained=False), [passage], [passage]
    )
    assert result["verbatim_support"] == supported


def test_exact_match_requires_correct_abstention():
    case = {"expected_sources": [], "expected_answer": "Not enough evidence."}
    result = score(case, Prediction(answer=case["expected_answer"], citations=[], abstained=False), [], [])
    assert result["exact_match"] == 0 and result["abstention_correct"] == 0
    assert result["retrieval_recall"] is None


def test_reference_run_is_real_and_has_no_token_claim():
    result = run_comparison(RunRequest())
    assert len(result["pipelines"]["candidate"]["rows"]) == 30
    for pipeline in result["pipelines"].values():
        assert pipeline["metrics"]["input_tokens"] is None
        assert pipeline["metrics"]["estimated_cost_usd"] == 0
        assert pipeline["metrics"]["errors"] == 0
        assert all(r["latency_ms"] >= 0 for r in pipeline["rows"])
    assert result["model"] == "extractive-v1"


def test_gate_rejects_regression_errors_and_latency():
    baseline = {"exact_match": 0.9, "retrieval_recall": 1, "errors": 0, "p95_latency_ms": 1}
    candidate = dict(baseline, exact_match=0.8, errors=1, p95_latency_ms=6000)
    assert not release_gate(baseline, candidate)["passed"]
    assert release_gate(baseline, baseline)["passed"]


def test_live_adapter_disabled_by_default():
    with pytest.raises(ValueError, match="Live inference"):
        run_comparison(RunRequest(provider="openai"), Settings(_env_file=None))


def test_openai_adapter_structured_request_excludes_labels():
    captured = {}

    def parse(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            output_parsed=Prediction(answer="Evidence", citations=["p1"], abstained=False),
            usage=SimpleNamespace(input_tokens=30, output_tokens=8),
        )

    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    result, usage = openai_predict(
        "Question?", [{"id": "p1", "text": "Evidence"}], client, "configured-model"
    )
    assert not captured["store"]
    assert captured["text_format"] is Prediction
    assert "expected_answer" not in str(captured["input"])
    assert usage == {"input_tokens": 30, "output_tokens": 8}
    assert result.answer == "Evidence"


@pytest.mark.parametrize("change", [{"top_k": 100}, {"threshold": -1}, {"retriever": "unknown"}])
def test_invalid_configuration_rejected(change):
    with pytest.raises(ValidationError):
        RunRequest(baseline={**BASELINE.model_dump(), **change})
