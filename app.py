from __future__ import annotations
import sys
from pathlib import Path
import json
import os
import streamlit as st
import pandas as pd
import sys, openai

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))

from gs_match_ai.ingest import load_l2_csv
from gs_match_ai.stats import compute_match_stats
from gs_match_ai.candidates import select_candidate_points
from gs_match_ai.openai_wrappers import generate_insights, answer_question
from gs_match_ai.utils import fmt_mmss, sha256_file

# Streamlit Cloud: pull secrets into env vars for the OpenAI SDK
if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "OPENAI_MODEL" not in os.environ and "OPENAI_MODEL" in st.secrets:
    os.environ["OPENAI_MODEL"] = st.secrets["OPENAI_MODEL"]

st.set_page_config(page_title="GameSmart Match AI Demo", layout="wide")

st.title("🎾 GameSmart — Match AI (MVP Demo)")
st.caption("Upload L2 CSV → compute Evidence Packet → generate Insight Objects → chat “Explain my match”. (No video playback yet)")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    max_candidates = st.slider("Candidate points (LLM evidence)", 20, 80, 60, 5)
    seed = st.number_input("Candidate sampling seed", min_value=0, value=7, step=1)
    st.markdown("---")
    st.write("Env needed: OPENAI_API_KEY")
    st.sidebar.caption(f"Python: {sys.version.split()[0]}")
    st.sidebar.caption(f"OpenAI SDK: {openai.__version__}")

for k, v in {"df": None, "evidence_packet": None, "candidates": None, "insights": None, "chat": []}.items():
    st.session_state.setdefault(k, v)

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Load L2 CSV")
    uploaded = st.file_uploader("Upload L2 CSV", type=["csv"])

    sample_dir = ROOT / "data" / "raw"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted([str(p) for p in sample_dir.glob("*.csv")])
    sample_choice = st.selectbox("...or pick a sample", ["(none)"] + samples)

    if st.button("Load CSV", type="primary"):
        try:
            if uploaded:
                tmp = ROOT / "data" / "processed" / "_tmp_upload.csv"
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(uploaded.getvalue())
                df = load_l2_csv(tmp)
                st.session_state.csv_sha = sha256_file(tmp)
                st.session_state.csv_name = uploaded.name
            elif sample_choice != "(none)":
                df = load_l2_csv(sample_choice)
                st.session_state.csv_sha = sha256_file(sample_choice)
                st.session_state.csv_name = Path(sample_choice).name
            else:
                st.warning("Upload a CSV or choose a sample.")
                df = None

            if df is not None:
                st.session_state.df = df
                st.session_state.evidence_packet = None
                st.session_state.candidates = None
                st.session_state.insights = None
                st.session_state.chat = []
                st.success(f"Loaded {len(df)} points: {st.session_state.csv_name}")
        except Exception as e:
            st.error(f"Load failed: {e}")

    if st.session_state.df is not None:
        st.dataframe(
            st.session_state.df[["point_idx","start_s","end_s","server","pt_won_by","end_type","rally_length","rally_summary"]].head(10),
            use_container_width=True
        )

with right:
    st.subheader("2) Build foundation artifacts")
    if st.session_state.df is None:
        st.info("Load a CSV first.")
    else:
        if st.button("Compute Evidence Packet"):
            stats = compute_match_stats(st.session_state.df)
            match_meta = {
                "csv_name": getattr(st.session_state, "csv_name", "upload.csv"),
                "csv_sha256": getattr(st.session_state, "csv_sha", ""),
                "players": stats["players"],
            }
            st.session_state.evidence_packet = {"match": match_meta, **stats}
            st.success("Evidence Packet computed.")

        if st.button("Select Candidate Points"):
            st.session_state.candidates = select_candidate_points(
                st.session_state.df, max_points=int(max_candidates), seed=int(seed)
            )
            st.success(f"Selected {len(st.session_state.candidates)} candidate points.")

        if st.button("Generate Insight Objects (LLM)", disabled=st.session_state.evidence_packet is None or st.session_state.candidates is None):
            try:
                st.session_state.insights = generate_insights(
                    st.session_state.evidence_packet["match"],
                    st.session_state.evidence_packet,
                    st.session_state.candidates,
                    model=model
                )
                st.success("Insights generated.")
            except Exception as e:
                st.error(f"LLM failed: {e}")

st.markdown("---")
tabs = st.tabs(["Stats", "Insights", "Chat", "Export"])

with tabs[0]:
    if st.session_state.evidence_packet is None:
        st.info("Compute Evidence Packet first.")
    else:
        st.json(st.session_state.evidence_packet, expanded=False)

with tabs[1]:
    if st.session_state.insights is None:
        st.info("Generate Insight Objects first.")
    else:
        ins = st.session_state.insights
        st.markdown("### Entry summary")
        for line in ins.get("entry_summary", []):
            st.write(f"- {line}")

        idx_to_point = {p["point_idx"]: p for p in (st.session_state.candidates or [])}

        st.markdown("### Insight Objects")
        for i, obj in enumerate(ins.get("insights", []), start=1):
            with st.expander(f"{i}. {obj['title']} (priority {obj['priority']:.2f})", expanded=(i<=3)):
                st.write(obj["summary"])
                st.markdown(f"**Coaching tip:** {obj['coaching_tip']}")
                st.markdown("**Evidence refs:** " + ", ".join(obj["evidence_refs"]))
                st.markdown("**Supporting points:** " + ", ".join(map(str, obj["supporting_points"])))

                st.markdown("**Timestamp evidence:**")
                for pid in obj["supporting_points"][:8]:
                    p = idx_to_point.get(pid)
                    if p:
                        st.write(f"- #{pid} [{fmt_mmss(p['start_s'])}–{fmt_mmss(p['end_s'])}] — {p['rally_summary']}")

with tabs[2]:
    if st.session_state.evidence_packet is None or st.session_state.insights is None or st.session_state.candidates is None:
        st.info("Compute Evidence + Candidates + Insights first.")
    else:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        q = st.chat_input("Ask: Why did I lose? How was my 2nd serve? What should I work on?")
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    ans = answer_question(
                        q,
                        st.session_state.evidence_packet["match"],
                        st.session_state.evidence_packet,
                        st.session_state.insights,
                        st.session_state.candidates,
                        model=model
                    )
                    st.markdown(ans)
                    st.session_state.chat.append({"role": "assistant", "content": ans})

with tabs[3]:
    if st.session_state.evidence_packet is not None:
        st.download_button(
            "Download evidence_packet.json",
            json.dumps(st.session_state.evidence_packet, indent=2, ensure_ascii=False),
            "evidence_packet.json",
            "application/json"
        )
    if st.session_state.candidates is not None:
        st.download_button(
            "Download candidate_points.json",
            json.dumps(st.session_state.candidates, indent=2, ensure_ascii=False),
            "candidate_points.json",
            "application/json"
        )
    if st.session_state.insights is not None:
        st.download_button(
            "Download insight_objects.json",
            json.dumps(st.session_state.insights, indent=2, ensure_ascii=False),
            "insight_objects.json",
            "application/json"
        )
