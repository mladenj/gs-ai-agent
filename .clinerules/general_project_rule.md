# GameSmart AI Agent Demo – Rules

## Must keep working
- Streamlit Cloud deploy must remain functional.
- app entrypoint stays `app.py`.

## Environment
- Use Python 3.11 + `.venv`
- Always run commands via `python -m ...` (ensures venv)

## Secrets
- Never commit secrets.
- Use Streamlit secrets or env vars only.
- Do NOT write/overwrite `.env` or `.streamlit/secrets.toml` unless explicitly asked.

## Dev loop
- After changes: `python -m streamlit run app.py` and fix any runtime errors.
- Keep dependencies minimal; if adding a dep, update requirements.txt.
