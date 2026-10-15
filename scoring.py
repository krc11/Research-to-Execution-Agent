"""Step 4 – Pure-code weighted scoring.  The model does NOT do the math."""

from __future__ import annotations

from typing import List

from schemas import (
    Criterion,
    CriterionPriority,
    CriterionScore,
    ScoredCandidate,
    VerifiedFinding,
)


def score_candidates(
    verified: List[VerifiedFinding],
    criteria: List[Criterion],
    candidates: List[str],
) -> List[ScoredCandidate]:
    """Return candidates sorted best-first by weighted score."""

    scored: List[ScoredCandidate] = []

    for candidate in candidates:
        cand_findings = [f for f in verified if f.candidate.lower() == candidate.lower()]
        criterion_scores: List[CriterionScore] = []
        total_weighted = 0.0
        max_weighted = 0.0
        must_have_met = True

        for crit in criteria:
            matching = [
                f for f in cand_findings
                if f.criterion.lower() == crit.name.lower()
            ]

            if matching:
                avg_rating = sum(f.rating for f in matching) / len(matching)
            else:
                avg_rating = 0.0

            weighted = avg_rating * crit.weight
            max_w = 10.0 * crit.weight
            total_weighted += weighted
            max_weighted += max_w

            if crit.priority == CriterionPriority.MUST_HAVE and avg_rating == 0:
                must_have_met = False

            criterion_scores.append(
                CriterionScore(
                    criterion=crit.name,
                    weight=crit.weight,
                    rating=round(avg_rating, 2),
                    weighted_score=round(weighted, 2),
                    finding_ids=[f.id for f in matching],
                )
            )

        percentage = (total_weighted / max_weighted * 100) if max_weighted > 0 else 0.0

        scored.append(
            ScoredCandidate(
                candidate=candidate,
                total_score=round(total_weighted, 2),
                max_possible=round(max_weighted, 2),
                percentage=round(percentage, 1),
                criterion_scores=criterion_scores,
                must_have_met=must_have_met,
            )
        )

    # Best score first; break ties by must_have_met
    scored.sort(key=lambda s: (s.must_have_met, s.total_score), reverse=True)
    return scored
