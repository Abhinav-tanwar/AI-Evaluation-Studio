from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    retriever: Literal["overlap", "bm25"]
    top_k: int = Field(default=3, ge=1, le=5)
    threshold: float = Field(default=0.35, ge=0, le=1)


BASELINE = PipelineConfig(name="Token overlap", retriever="overlap", top_k=1, threshold=0.25)
CANDIDATE = PipelineConfig(name="BM25 retrieval", retriever="bm25", top_k=3, threshold=0.45)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="Retrieval comparison", min_length=1, max_length=80)
    split: Literal["development", "holdout"] = "development"
    provider: Literal["reference", "openai"] = "reference"
    baseline: PipelineConfig = BASELINE
    candidate: PipelineConfig = CANDIDATE


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(max_length=5000)
    citations: list[str] = Field(max_length=10)
    abstained: bool


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^q\d{2}$")
    verdict: Literal["correct", "incorrect", "needs review"]
    note: str = Field(default="", max_length=2000)
