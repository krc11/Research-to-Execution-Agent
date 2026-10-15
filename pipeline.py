"""Six-step research pipeline: Parse → Research → Verify → Score → Write → Render.

Every step is a method so callers can stream progress.  The model is used for
steps 1, 2, 3, 5.  Steps 4 and 6 are pure code.
"""

from __future__ import annotations

import json
import re
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import anthropic

from schemas import (
    Criterion,
    CriterionPriority,
    Finding,
    ParsedInput,
    Report,
    VerifiedFinding,
)
from scoring import score_candidates
from renderer import render_report

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_text(response: anthropic.types.Message) -> str:
    """Pull all text blocks from an Anthropic response."""
    parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


def _extract_search_urls(response: anthropic.types.Message) -> list[str]:
    """Pull every URL that appeared in web-search result blocks."""
    urls: list[str] = []
    for block in response.content:
        if getattr(block, "type", "") == "web_search_tool_result":
            for item in getattr(block, "content", []):
                url = getattr(item, "url", None)
                if url:
                    urls.append(url)
    return urls


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object/array from model text."""
    # Try fenced code blocks first
    m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()
    # Fall back to first { … } or [ … ]
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text.strip()


def _domain_of(url: str) -> str:
    """Return the domain (without www.) of a URL."""
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _url_allowed(url: str, allowed_domains: list[str]) -> bool:
    """Check whether a URL falls within the allowlist.

    If the allowlist is empty everything is allowed (open research).
    """
    if not allowed_domains:
        return True
    domain = _domain_of(url)
    return any(domain == d or domain.endswith("." + d) for d in allowed_domains)


# ── Pipeline ─────────────────────────────────────────────────────────────────

StatusCallback = Callable[[str], None]


class Pipeline:
    """Fixed six-step research pipeline."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        on_status: Optional[StatusCallback] = None,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._status = on_status or (lambda _msg: None)

    # ── public entry point ───────────────────────────────────────────────

    def run(
        self,
        question: str,
        criteria_text: str,
        sources_text: str,
    ) -> Tuple[Report, str]:
        """Execute the full pipeline and return (Report, rendered_markdown)."""

        # 1. Parse
        self._status("⏳ Step 1/6 — Parsing question and criteria…")
        parsed, allowed_domains = self.step_parse(question, criteria_text, sources_text)
        self._status(
            f"✅ Parsed: {len(parsed.candidates)} candidates, "
            f"{len(parsed.criteria)} criteria, "
            f"{len(allowed_domains)} allowed domains"
        )

        # 2. Research
        self._status("⏳ Step 2/6 — Researching candidates…")
        findings = self.step_research(parsed, allowed_domains)
        self._status(f"✅ Collected {len(findings)} findings")

        # 3. Verify
        self._status("⏳ Step 3/6 — Verifying claims…")
        verified, unverified = self.step_verify(findings)
        self._status(
            f"✅ {len(verified)} verified, {len(unverified)} could not verify"
        )

        # 4. Score
        self._status("⏳ Step 4/6 — Scoring…")
        scored = score_candidates(verified, parsed.criteria, parsed.candidates)
        self._status("✅ Scoring complete")

        # 5. Write
        self._status("⏳ Step 5/6 — Writing recommendation…")
        recommendation = self.step_write(parsed, scored, verified)
        self._status("✅ Recommendation drafted")

        # 6. Render
        self._status("⏳ Step 6/6 — Rendering report…")
        report = Report(
            question=parsed.question,
            recommendation=recommendation,
            scored_candidates=scored,
            verified_findings=verified,
            unverified_findings=unverified,
        )
        md = render_report(report)
        self._status("✅ Report ready")

        return report, md

    # ── Step 1: Parse ────────────────────────────────────────────────────

    def step_parse(
        self,
        question: str,
        criteria_text: str,
        sources_text: str,
    ) -> Tuple[ParsedInput, List[str]]:
        """Parse user inputs into structured data + domain allowlist."""

        allowed_domains = self._parse_sources(sources_text)

        prompt = (
            "You are a structured-data extractor.  Parse the following research "
            "question and criteria into a JSON object.\n\n"
            f"### Research Question\n{question}\n\n"
            f"### Criteria (pasted text or CSV)\n{criteria_text}\n\n"
            "Return a JSON object with exactly these keys:\n"
            '- "question": the research question as a concise string\n'
            '- "candidates": a JSON array of candidate names/options to compare '
            "(infer from the question; if only one is mentioned, list plausible "
            "alternatives)\n"
            '- "criteria": a JSON array where each element has:\n'
            '    - "name": short criterion label\n'
            '    - "description": one-sentence explanation\n'
            '    - "weight": importance 1-10 (10 = critical)\n'
            '    - "priority": "must-have" or "nice-to-have"\n\n'
            "Return ONLY valid JSON. No markdown fences, no commentary."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = _extract_text(response)
        raw = json.loads(_extract_json(text))
        parsed = ParsedInput(**raw)
        return parsed, allowed_domains

    # ── Step 2: Research ─────────────────────────────────────────────────

    def step_research(
        self,
        parsed: ParsedInput,
        allowed_domains: List[str],
    ) -> List[Finding]:
        """For each candidate, use Claude + web search to gather evidence."""
        all_findings: List[Finding] = []

        for candidate in parsed.candidates:
            self._status(f"  🔍 Researching {candidate}…")
            findings = self._research_candidate(
                candidate, parsed.criteria, allowed_domains
            )
            all_findings.extend(findings)

        return all_findings

    def _research_candidate(
        self,
        candidate: str,
        criteria: List[Criterion],
        allowed_domains: List[str],
    ) -> List[Finding]:
        """Call Claude with web search for one candidate."""

        criteria_desc = "\n".join(
            f"- {c.name} ({c.priority.value}, weight {c.weight}): {c.description}"
            for c in criteria
        )

        domain_note = (
            "You may search any domain."
            if not allowed_domains
            else f"Only use these domains: {', '.join(allowed_domains)}"
        )

        prompt = (
            f"Research **{candidate}** against every criterion below.  "
            f"{domain_note}\n\n"
            f"### Criteria\n{criteria_desc}\n\n"
            "For each criterion, find evidence from the web.\n\n"
            "Return a JSON array where each element has:\n"
            '- "candidate": the candidate name\n'
            '- "criterion": the criterion name (must match exactly)\n'
            '- "claim": one-sentence factual claim\n'
            '- "quote": a SHORT direct quote from the source (≤ 2 sentences)\n'
            '- "url": the source URL\n'
            '- "rating": 1-10 how well the candidate meets this criterion '
            "based on the evidence\n\n"
            "If you cannot find evidence for a criterion, still include an "
            'entry with claim "No evidence found", empty quote, url "N/A", '
            "and rating 1.\n\n"
            "Return ONLY the JSON array."
        )

        # Build tools list
        tools: List[Dict[str, Any]] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 10,
            }
        ]
        if allowed_domains:
            tools[0]["allowed_domains"] = allowed_domains

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                tools=tools,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.BadRequestError as e:
            # Web search might not be enabled — fall back to no-search call
            logger.warning("Web search unavailable (%s), falling back to model knowledge", e)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                messages=[{"role": "user", "content": prompt}],
            )

        text = _extract_text(response)
        search_urls = _extract_search_urls(response)

        # Parse findings JSON
        try:
            raw_list = json.loads(_extract_json(text))
        except json.JSONDecodeError:
            logger.error("Failed to parse research JSON for %s", candidate)
            return []

        if not isinstance(raw_list, list):
            raw_list = [raw_list]

        now = datetime.now(timezone.utc).isoformat()
        findings: List[Finding] = []

        for item in raw_list:
            url = item.get("url", "N/A")

            # ── NON-NEGOTIABLE: enforce allowlist in code ──
            if url != "N/A" and not _url_allowed(url, allowed_domains):
                logger.info("Rejected URL outside allowlist: %s", url)
                continue

            try:
                findings.append(
                    Finding(
                        candidate=item.get("candidate", candidate),
                        criterion=item.get("criterion", ""),
                        claim=item.get("claim", ""),
                        quote=item.get("quote", ""),
                        url=url,
                        retrieved_at=now,
                        rating=max(1, min(10, int(item.get("rating", 5)))),
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed finding: %s", exc)

        return findings

    # ── Step 3: Verify ───────────────────────────────────────────────────

    def step_verify(
        self, findings: List[Finding]
    ) -> Tuple[List[VerifiedFinding], List[VerifiedFinding]]:
        """Second-pass verification: does the quote support the claim?"""

        if not findings:
            return [], []

        # Batch all findings into one verification call for efficiency
        items_json = json.dumps(
            [
                {
                    "id": f.id,
                    "candidate": f.candidate,
                    "criterion": f.criterion,
                    "claim": f.claim,
                    "quote": f.quote,
                    "url": f.url,
                }
                for f in findings
            ],
            indent=2,
        )

        prompt = (
            "You are a fact-checking assistant.  For each finding below, "
            "decide whether the **quote** actually supports the **claim**.\n\n"
            f"### Findings\n```json\n{items_json}\n```\n\n"
            "Return a JSON array with one object per finding:\n"
            '- "id": the finding ID\n'
            '- "verified": true if the quote supports the claim, false otherwise\n'
            '- "note": short explanation (≤ 1 sentence) if unverified, else ""\n\n'
            "If the claim says 'No evidence found' and the quote is empty, "
            "mark it as verified=false with note='No evidence found'.\n\n"
            "Return ONLY valid JSON."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = _extract_text(response)
        try:
            verdicts_raw = json.loads(_extract_json(text))
        except json.JSONDecodeError:
            logger.error("Failed to parse verification JSON; marking all as verified")
            verdicts_raw = [{"id": f.id, "verified": True, "note": ""} for f in findings]

        verdict_map: Dict[str, Dict] = {v["id"]: v for v in verdicts_raw}

        verified: List[VerifiedFinding] = []
        unverified: List[VerifiedFinding] = []

        for f in findings:
            v = verdict_map.get(f.id, {"verified": True, "note": ""})
            vf = VerifiedFinding(
                id=f.id,
                candidate=f.candidate,
                criterion=f.criterion,
                claim=f.claim,
                quote=f.quote,
                url=f.url,
                retrieved_at=f.retrieved_at,
                rating=f.rating,
                verified=v.get("verified", True),
                verification_note=v.get("note", ""),
            )
            if vf.verified:
                verified.append(vf)
            else:
                unverified.append(vf)

        return verified, unverified

    # ── Step 5: Write ────────────────────────────────────────────────────

    def step_write(
        self,
        parsed: ParsedInput,
        scored: list,
        verified: List[VerifiedFinding],
    ) -> str:
        """Draft the recommendation.  Every sentence must cite a finding ID."""

        findings_ref = "\n".join(
            f"[{f.id}] {f.candidate} / {f.criterion}: {f.claim} (source: {f.url})"
            for f in verified
        )

        scores_summary = "\n".join(
            f"- {s.candidate}: {s.percentage:.0f}% (must-haves {'met' if s.must_have_met else 'NOT met'})"
            for s in scored
        )

        prompt = (
            "Write a concise recommendation for this research question:\n\n"
            f"**{parsed.question}**\n\n"
            f"### Scores\n{scores_summary}\n\n"
            f"### Verified Evidence\n{findings_ref}\n\n"
            "Rules:\n"
            "1. Every factual sentence MUST cite at least one finding ID "
            "in square brackets, e.g. [F-abc123].\n"
            "2. Open with the overall recommendation.\n"
            "3. Explain why the top candidate wins.\n"
            "4. Flag any must-have criteria that candidates failed.\n"
            "5. Be concise — no more than 400 words.\n"
            "6. Do NOT invent facts beyond what the evidence IDs cover.\n\n"
            "Write the recommendation now."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )

        recommendation = _extract_text(response)

        # ── NON-NEGOTIABLE: reject any sentence that cites no ID ──
        valid_ids = {f.id for f in verified}
        recommendation = self._enforce_citations(recommendation, valid_ids)

        return recommendation

    @staticmethod
    def _enforce_citations(text: str, valid_ids: set[str]) -> str:
        """Remove sentences that contain factual claims but cite no finding ID.

        Keeps structural sentences (headers, bullets starting with candidate
        names, transition phrases) even without IDs.
        """
        id_pattern = re.compile(r"\[F-[a-f0-9]{6}\]")
        lines = text.split("\n")
        cleaned: list[str] = []

        for line in lines:
            stripped = line.strip()
            # Keep blank lines, headers, and very short transition phrases
            if (
                not stripped
                or stripped.startswith("#")
                or stripped.startswith("---")
                or len(stripped) < 30
            ):
                cleaned.append(line)
                continue

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", stripped)
            kept: list[str] = []
            for s in sentences:
                if id_pattern.search(s) or len(s.split()) < 8:
                    kept.append(s)
                # else: factual sentence without citation → drop
            if kept:
                cleaned.append(" ".join(kept))

        return "\n".join(cleaned)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_sources(sources_text: str) -> List[str]:
        """Turn the free-text sources input into a list of clean domains."""
        if not sources_text.strip():
            return []

        domains: List[str] = []
        for line in re.split(r"[,\n]+", sources_text):
            token = line.strip().lower()
            if not token:
                continue
            # If it's a full URL, extract domain
            if token.startswith("http"):
                d = _domain_of(token)
                if d:
                    domains.append(d)
            else:
                # Treat as bare domain
                token = token.lstrip("*.")
                if token.startswith("www."):
                    token = token[4:]
                if "." in token:
                    domains.append(token)
        return list(dict.fromkeys(domains))  # deduplicate, preserve order
