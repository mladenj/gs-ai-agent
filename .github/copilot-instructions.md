# GameSmart Match AI Agent Instructions

## Project Overview
A Streamlit MVP for analyzing tennis match L2 (point-level) CSV data. The pipeline ingests match data → computes statistics → selects key points → generates AI-driven insights via OpenAI.

**Core flow:** CSV → normalized DataFrame → Evidence Packet (stats) + Candidate Points (filtered high-signal points) → LLM generates Insight Objects → users query insights in chat.

## Architecture & Data Flow

### 1. **Data Ingestion (`src/gs_match_ai/ingest.py`)**
- **Entry point:** `load_l2_csv(path)` → returns normalized DataFrame
- **REQUIRED_COLS:** `p1_id`, `p2_id`, `p1_fullname`, `p2_fullname`, `server`, `pt_won_by`, `vid_second`, `end`, `rally_length`, `rally_summary`, `rally_desc`, `discard_point`
- **Normalization:** Parses serve strings ("down the t", "wide", etc.) → `final_serve_dir`, fault detection, `start_s`/`end_s` timestamps
- **Critical constraint:** CSV column names must match exactly; adjust `REQUIRED_COLS` if new data sources differ

### 2. **Evidence Packet (`src/gs_match_ai/stats.py`)**
- `compute_match_stats(df)` computes detailed KPIs: serve/return stats by player, win percentages, double faults, serve direction win rates
- Excludes rows where `discard_point==True`
- Output: nested dict with `players`, `totals`, `kpis` (serve/return blocks)
- **Pattern:** Uses `_pct()` helper to round percentages to 2 decimals

### 3. **Candidate Point Selection (`src/gs_match_ai/candidates.py`)**
- `select_candidate_points(df, max_points=60, seed=7)` prioritizes high-signal points
- **Selection strategy:** breakpoints → setpoints → matchpoints → gamepoints (if columns exist) → top 20 winners/errors → top 10 rally_length → 10 random
- **Seeded randomness** for reproducibility; respects `discard_point` filter
- Returns list of dicts with `point_idx`, `start_s`, `end_s`, `server`, `pt_won_by`, `end_type`, rally metadata
- **Why:** LLM context limits; these points are evidence "anchors" for insights

### 4. **LLM Integration (`src/gs_match_ai/openai_wrappers.py`)**
- **`generate_insights()`** → uses OpenAI Structured Outputs (JSON schema mode)
  - Loads system prompt from `prompts/insight_writer.md`
  - Input: match_metadata + evidence_packet + candidate_points (full)
  - Output validates against `INSIGHTS_JSON_SCHEMA` (5+ insights, with topic/title/summary/coaching_tip/priority/evidence_refs/supporting_points)
  - Model default: `gpt-4o-mini` (via env var)
  
- **`answer_question()`** → chat query answering
  - Loads `prompts/chat_answerer.md`
  - Input: question + metadata + evidence + insights + slim_points (point_idx/start_s/end_s/rally_summary/end_type only)
  - Plain text response; used for conversational follow-ups

### 5. **UI (`app.py` — Streamlit)**
- **Session state keys:** `df`, `evidence_packet`, `candidates`, `insights`, `chat` (list of Q&A tuples)
- **Workflow:** Upload CSV → Compute Evidence Packet → Select Candidates → Generate Insights → Chat
- **Settings sidebar:** model name, max candidates (20–80, default 60), seed (for reproducibility)
- **Tabs:** Stats (JSON), Insights (parsed objects + supporting point timestamps), Chat, Export
- **Sample CSVs:** stored in `data/raw/`; uploaded files written to `data/processed/_tmp_upload.csv`

## Key Conventions & Patterns

1. **Player IDs as strings:** `p1_id`, `p2_id` are treated as strings throughout (e.g., `str(r["server"])`)
2. **NaN handling:** Extensive `.fillna()`, `.isna()` checks; `pd.isna()` for numeric validation
3. **Serve parsing:** Complex regex on serve text → `final_serve_dir` (T/BODY/WIDE/UNKNOWN)
4. **Discard filter:** Applied in stats/candidates but NOT at ingest; allows downstream flexibility
5. **Point indexing:** 1-based `point_idx` (column) vs. 0-based DataFrame index; always use `point_idx` in output
6. **Prompt templates:** Loaded as markdown from `prompts/` directory; referenced by relative path in wrappers
7. **Seed for determinism:** Candidate selection uses `np.random.default_rng(seed)` for reproducible sampling

## Development & Workflows

**Setup:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"  # optional; defaults in code
```

**Run Streamlit app:**
```bash
streamlit run app.py
```

**Run pipeline (CLI, non-interactive):**
```bash
python scripts/run_pipeline.py --csv data/raw/sample.csv --out data/processed/
```
Outputs: `evidence_packet.json`, `candidate_points.json`

**Testing/debugging:**
- Use sample CSVs in `data/raw/` for reproducible testing
- Set seed consistently for deterministic candidate selection
- Print JSON outputs to understand schema validation failures

## External Dependencies & Constraints

- **OpenAI SDK:** `responses.create()` uses new API; ensure version ≥ 2.15.0
- **Streamlit secrets:** Falls back to env vars if `st.secrets["OPENAI_API_KEY"]` not available
- **CSV schema:** Rigid; missing columns raise `ValueError`; new data sources may need `REQUIRED_COLS` extension
- **Structured Outputs:** Requires `strict=True` mode; schema violations cause LLM retries

## Common Extension Points

1. **New stat KPI:** Add column to `serve_block()` or `return_block()` in `stats.py`
2. **New candidate selection heuristic:** Add prioritization in `select_candidate_points()`
3. **Alternative LLM:** Swap OpenAI client in `openai_wrappers.py`
4. **New insight topics:** Update `INSIGHTS_JSON_SCHEMA` properties
5. **CSV format variation:** Extend `REQUIRED_COLS` and `normalize_points()` parsing logic
