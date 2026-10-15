"""Pydantic schemas for every stage of the six-step pipeline."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── helpers ──────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finding_id() -> str:
    return f"F-{uuid.uuid4().hex[:6]}"


# ── Step 1 – Parse ──────────────────────────────────────────────────────────

class CriterionPriority(str, Enum):
    MUST_HAVE = "must-have"
    NICE_TO_HAVE = "nice-to-have"


class Criterion(BaseModel):
    name: str
    description: str = ""
    weight: float = Field(default=5.0, ge=0.0, le=10.0)
    priority: CriterionPriority = CriterionPriority.NICE_TO_HAVE


class ParsedInput(BaseModel):
    question: str
    candidates: List[str]
    criteria: List[Criterion]


# ── Step 2 – Research ───────────────────────────────────────────────────────

class Finding(BaseModel):
    id: str = Field(default_factory=_finding_id)
    candidate: str
    criterion: str
    claim: str
    quote: str
    url: str
    retrieved_at: str = Field(default_factory=_utcnow)
    rating: int = Field(default=5, ge=1, le=10, description="1-10 how well the candidate meets this criterion based on evidence")


# ── Step 3 – Verify ─────────────────────────────────────────────────────────

class VerifiedFinding(BaseModel):
    id: str
    candidate: str
    criterion: str
    claim: str
    quote: str
    url: str
    retrieved_at: str
    rating: int
    verified: bool = True
    verification_note: str = ""


# ── Step 4 – Score ──────────────────────────────────────────────────────────

class CriterionScore(BaseModel):
    criterion: str
    weight: float
    rating: float
    weighted_score: float
    finding_ids: List[str]


class ScoredCandidate(BaseModel):
    candidate: str
    total_score: float
    max_possible: float
    percentage: float
    criterion_scores: List[CriterionScore]
    must_have_met: bool


# ── Step 6 – Full Report ────────────────────────────────────────────────────

class Report(BaseModel):
    question: str
    recommendation: str
    scored_candidates: List[ScoredCandidate]
    verified_findings: List[VerifiedFinding]
    unverified_findings: List[VerifiedFinding]
    generated_at: str = Field(default_factory=_utcnow)
