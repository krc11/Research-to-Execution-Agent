# 🔬 Research-to-Execution Agent v1

A six-step research pipeline that takes a question, criteria, and permitted
sources and produces a source-linked recommendation report.

## What it does

Given:
- A **research question** (e.g. "Which CRM should we use?")
- **Criteria** (pasted text or CSV — each gets a weight and must-have / nice-to-have flag)
- **Permitted sources** (domain allowlist — enforced in code, not just the prompt)

It returns **one report** containing:
1. **Recommendation** — every claim cites an evidence ID
2. **Comparison table** — weighted scores per criterion
3. **Evidence ledger** — claim, quote, URL, date retrieved
4. **"Could not verify" section** — claims whose quotes didn't support them

## Architecture — fixed 6-step pipeline

| Step | What happens | Model? |
|------|-------------|--------|
| 1. **Parse** | Question + criteria → structured JSON | ✅ |
| 2. **Research** | Per candidate, Claude + web search on allowed domains | ✅ |
| 3. **Verify** | Second pass: does the quote support the claim? | ✅ |
| 4. **Score** | Weighted scoring in plain Python (model doesn't do math) | ❌ |
| 5. **Write** | Draft recommendation; code strips un-cited sentences | ✅ |
| 6. **Render** | Markdown report with tables and download button | ❌ |

## Non-negotiable rules

1. **Allowlist enforced in code** — any URL outside it is rejected, regardless of what the model returns.
2. **No claim without an evidence ID** — "Not found" is a valid result.
3. **Every piece of evidence is timestamped.**

## Quick start (local)

```bash
# 1. Clone & install
pip install -r requirements.txt

# 2. Add your API key
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml → set ANTHROPIC_API_KEY and PASSCODE

# 3. Run
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select repo & `app.py`.
3. In **Advanced settings → Secrets**, paste:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   PASSCODE = "your-passcode"
   ```
4. Deploy.

## Project structure

```
├── app.py              # Streamlit UI (passcode gate, inputs, report display)
├── pipeline.py         # 6-step pipeline orchestrator
├── schemas.py          # Pydantic models for every stage
├── scoring.py          # Pure-code weighted scoring (step 4)
├── renderer.py         # Markdown report renderer (step 6)
├── requirements.txt
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── README.md
```

## Fallback: no web search

If your Anthropic account doesn't have web search enabled, the pipeline
catches the error and falls back to model knowledge (no live web data).
To use live data in that case, provide specific URLs in the sources field
and the pipeline will note the limitation.

## Model

Default: `claude-sonnet-4-20250514`.  Override via the `MODEL` secret.
