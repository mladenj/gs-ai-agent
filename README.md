# GameSmart Match AI Demo (Streamlit)

MVP: Upload an L2 match CSV → compute Evidence Packet → generate AI Insight Objects → chat “Explain my match”.
Video playback is skipped; insights include timestamps (start_s/end_s).

## Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="YOUR_KEY"        # Windows PowerShell: $env:OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"       # optional

streamlit run app.py

## Notes
- Artifacts can be exported from the app (JSON downloads).
- If your L2 CSV column names differ, adjust src/gs_match_ai/ingest.py REQUIRED_COLS.
