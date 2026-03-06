from __future__ import annotations
import sys
from pathlib import Path
import json
import os
import streamlit as st
import pandas as pd
import openai

ROOT = Path(__file__).parent
sys.path.append(str(ROOT / "src"))

from gs_match_ai.ingest import load_l2_csv
from gs_match_ai.stats import compute_match_stats
from gs_match_ai.candidates import select_candidate_points
from gs_match_ai.openai_wrappers import (
    generate_insights,
    answer_question,
    generate_multi_match_insights,
    answer_question_multi,
)
from gs_match_ai.utils import fmt_mmss, sha256_file

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
def read_openai_key():
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None

api_key = read_openai_key()
if not api_key:
    st.error("Missing OPENAI_API_KEY. Set it as an environment variable or in .streamlit/secrets.toml")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="GameSmart Match AI Demo", layout="wide")
st.title("🎾 GameSmart — Match AI (MVP Demo)")
st.caption("Upload one or more L2 CSVs → analyse single matches or compare across matches → chat with the AI coach.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value="gpt-4o-mini")
    max_candidates = st.slider("Candidate points (LLM evidence)", 20, 80, 60, 5)
    seed = st.number_input("Candidate sampling seed", min_value=0, value=7, step=1)
    st.markdown("---")
    st.write("Env needed: OPENAI_API_KEY")
    st.sidebar.caption(f"Python: {sys.version.split()[0]}")
    st.sidebar.caption(f"OpenAI SDK: {openai.__version__}")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
# matches: dict[filename -> {df, evidence_packet, candidates, insights, csv_sha}]
# chat: dict[filename -> list of {role, content}]
# multi_chat: list of {role, content}
# multi_insights: dict | None
# analysis_mode: "single" | "multi"
# selected_match: str
# selected_matches: list[str]

for k, v in {
    "matches": {},
    "chat": {},
    "multi_chat": [],
    "multi_insights": None,
    "multi_focus_player": None,
    "analysis_mode": "single",
    "selected_match": None,
    "selected_matches": [],
}.items():
    st.session_state.setdefault(k, v)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _badge(ok: bool) -> str:
    return "✅" if ok else "⬜"

def _match_label(idx: int) -> str:
    return f"match_{idx + 1}"

def _common_players(match_names: list[str]) -> dict[str, str]:
    """Return {player_id: player_name} for players present in ALL given matches."""
    if not match_names:
        return {}
    sets = []
    id_to_name: dict[str, str] = {}
    for name in match_names:
        df = st.session_state["matches"][name]["df"]
        row = df.iloc[0]
        p1_id = str(row["p1_id"])
        p2_id = str(row["p2_id"])
        p1_name = str(row["p1_fullname"]).strip()
        p2_name = str(row["p2_fullname"]).strip()
        sets.append({p1_id, p2_id})
        id_to_name[p1_id] = p1_name
        id_to_name[p2_id] = p2_name
    common_ids = sets[0].intersection(*sets[1:])
    return {pid: id_to_name[pid] for pid in sorted(common_ids)}

# ---------------------------------------------------------------------------
# Section 1: Load Matches
# ---------------------------------------------------------------------------
st.markdown("## 1 · Load Matches")

with st.expander("Upload CSVs or pick samples", expanded=True):
    uploaded_files = st.file_uploader(
        "Upload one or more L2 CSV files",
        type=["csv"],
        accept_multiple_files=True,
    )

    sample_dir = ROOT / "data" / "raw"
    sample_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted([str(p) for p in sample_dir.glob("*.csv")])
    sample_choice = st.selectbox("...or add a sample file", ["(none)"] + samples)

    if st.button("Load / Refresh CSVs", type="primary"):
        tmp_dir = ROOT / "data" / "processed"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        sources: list[tuple[str, Path]] = []  # (display_name, path)

        for uf in (uploaded_files or []):
            tmp = tmp_dir / f"_upload_{uf.name}"
            tmp.write_bytes(uf.getvalue())
            sources.append((uf.name, tmp))

        if sample_choice != "(none)":
            sp = Path(sample_choice)
            sources.append((sp.name, sp))

        if not sources:
            st.warning("Upload at least one CSV or choose a sample.")
        else:
            loaded_count = 0
            for name, path in sources:
                if name in st.session_state["matches"]:
                    st.info(f"'{name}' already loaded — skipping. Remove it first to reload.")
                    continue
                try:
                    df = load_l2_csv(path)
                    st.session_state["matches"][name] = {
                        "df": df,
                        "evidence_packet": None,
                        "candidates": None,
                        "insights": None,
                        "csv_sha": sha256_file(path),
                    }
                    st.session_state["chat"].setdefault(name, [])
                    loaded_count += 1
                    st.success(f"✅ Loaded '{name}': {len(df)} points.")
                except Exception as e:
                    st.error(f"Failed to load '{name}': {e}")

    # Table of loaded matches
    if st.session_state["matches"]:
        st.markdown("### Loaded matches")
        rows = []
        for name, m in st.session_state["matches"].items():
            rows.append({
                "File": name,
                "Points": len(m["df"]) if m["df"] is not None else "—",
                "Evidence": _badge(m["evidence_packet"] is not None),
                "Candidates": _badge(m["candidates"] is not None),
                "Insights": _badge(m["insights"] is not None),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Remove a match
        remove_choice = st.selectbox(
            "Remove a match from session",
            ["(keep all)"] + list(st.session_state["matches"].keys()),
        )
        if st.button("Remove selected match") and remove_choice != "(keep all)":
            st.session_state["matches"].pop(remove_choice, None)
            st.session_state["chat"].pop(remove_choice, None)
            # Reset multi insights if they involved this match
            st.session_state["multi_insights"] = None
            st.session_state["multi_chat"] = []
            if st.session_state["selected_match"] == remove_choice:
                st.session_state["selected_match"] = None
            if remove_choice in st.session_state["selected_matches"]:
                st.session_state["selected_matches"].remove(remove_choice)
            st.rerun()

# ---------------------------------------------------------------------------
# Section 2: Analysis Mode
# ---------------------------------------------------------------------------
st.markdown("## 2 · Analysis Mode")

match_names = list(st.session_state["matches"].keys())

if not match_names:
    st.info("Load at least one match above to continue.")
    st.stop()

mode_col, pick_col = st.columns([1, 2])

with mode_col:
    analysis_mode = st.radio(
        "Mode",
        ["Single Match", "Multi-Match"],
        index=0 if st.session_state["analysis_mode"] == "single" else 1,
        horizontal=True,
    )
    st.session_state["analysis_mode"] = "single" if analysis_mode == "Single Match" else "multi"

with pick_col:
    if st.session_state["analysis_mode"] == "single":
        default_idx = 0
        if st.session_state["selected_match"] in match_names:
            default_idx = match_names.index(st.session_state["selected_match"])
        selected_match = st.selectbox("Choose match", match_names, index=default_idx)
        st.session_state["selected_match"] = selected_match
    else:
        if len(match_names) < 2:
            st.warning("Multi-Match mode requires at least 2 loaded matches.")
            st.session_state["analysis_mode"] = "single"
        else:
            prev = [m for m in st.session_state["selected_matches"] if m in match_names]
            default_sel = prev if prev else match_names[:2]
            selected_matches = st.multiselect(
                "Choose matches to compare (2 or more)",
                match_names,
                default=default_sel,
            )
            st.session_state["selected_matches"] = selected_matches
            if len(selected_matches) < 2:
                st.warning("Select at least 2 matches for multi-match analysis.")
            else:
                # Detect players common to all selected matches
                common = _common_players(selected_matches)
                if not common:
                    st.error(
                        "⚠️ No player appears in all selected matches. "
                        "Multi-match analysis requires at least one shared player. "
                        "Select matches that share a player."
                    )
                    st.session_state["multi_focus_player"] = None
                else:
                    id_list = list(common.keys())
                    name_list = [common[pid] for pid in id_list]
                    # Keep previous selection if still valid
                    prev_fp = st.session_state.get("multi_focus_player") or {}
                    prev_idx = 0
                    if prev_fp.get("id") in id_list:
                        prev_idx = id_list.index(prev_fp["id"])
                    chosen_idx = st.selectbox(
                        "Focus player (appears in all selected matches)",
                        range(len(name_list)),
                        format_func=lambda i: name_list[i],
                        index=prev_idx,
                    )
                    st.session_state["multi_focus_player"] = {
                        "id": id_list[chosen_idx],
                        "name": name_list[chosen_idx],
                    }
                    st.caption(f"🎯 Analysis will focus on **{name_list[chosen_idx]}**")

# ---------------------------------------------------------------------------
# Section 3: Build Artifacts
# ---------------------------------------------------------------------------
st.markdown("## 3 · Build Artifacts")

mode = st.session_state["analysis_mode"]

if mode == "single":
    target_name = st.session_state["selected_match"]
    if not target_name or target_name not in st.session_state["matches"]:
        st.info("Select a match above.")
    else:
        m = st.session_state["matches"][target_name]

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Compute Evidence Packet", use_container_width=True):
                stats = compute_match_stats(m["df"])
                match_meta = {
                    "csv_name": target_name,
                    "csv_sha256": m["csv_sha"],
                    "players": stats["players"],
                }
                m["evidence_packet"] = {"match": match_meta, **stats}
                st.success("Evidence Packet computed.")

        with col2:
            if st.button("Select Candidate Points", use_container_width=True):
                m["candidates"] = select_candidate_points(
                    m["df"], max_points=int(max_candidates), seed=int(seed)
                )
                st.success(f"Selected {len(m['candidates'])} candidate points.")

        with col3:
            disabled = m["evidence_packet"] is None or m["candidates"] is None
            if st.button("Generate Insights (LLM)", disabled=disabled, use_container_width=True):
                try:
                    m["insights"] = generate_insights(
                        m["evidence_packet"]["match"],
                        m["evidence_packet"],
                        m["candidates"],
                        model=model,
                    )
                    st.success("Insights generated.")
                except Exception as e:
                    st.error(f"LLM failed: {e}")

        # Preview raw points
        with st.expander("Preview points (first 10)"):
            st.dataframe(
                m["df"][["point_idx","start_s","end_s","server","pt_won_by","end_type","rally_length","rally_summary"]].head(10),
                use_container_width=True,
            )

else:  # multi
    sel = st.session_state["selected_matches"]
    if len(sel) < 2:
        st.info("Select at least 2 matches above.")
    else:
        st.markdown(f"**Selected:** {', '.join(sel)}")

        # Per-match artifact buttons
        for name in sel:
            m = st.session_state["matches"][name]
            with st.expander(f"Artifacts for **{name}**  {_badge(m['evidence_packet'] is not None)} evidence  {_badge(m['candidates'] is not None)} candidates  {_badge(m['insights'] is not None)} insights"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button(f"Evidence Packet", key=f"ep_{name}", use_container_width=True):
                        stats = compute_match_stats(m["df"])
                        match_meta = {
                            "csv_name": name,
                            "csv_sha256": m["csv_sha"],
                            "players": stats["players"],
                        }
                        m["evidence_packet"] = {"match": match_meta, **stats}
                        st.success(f"Done.")
                with c2:
                    if st.button(f"Candidate Points", key=f"cp_{name}", use_container_width=True):
                        m["candidates"] = select_candidate_points(
                            m["df"], max_points=int(max_candidates), seed=int(seed)
                        )
                        st.success(f"Done: {len(m['candidates'])} pts.")
                with c3:
                    dis = m["evidence_packet"] is None or m["candidates"] is None
                    if st.button(f"Single-Match Insights", key=f"si_{name}", disabled=dis, use_container_width=True):
                        try:
                            m["insights"] = generate_insights(
                                m["evidence_packet"]["match"],
                                m["evidence_packet"],
                                m["candidates"],
                                model=model,
                            )
                            st.success(f"Done.")
                        except Exception as e:
                            st.error(f"LLM failed: {e}")

        st.markdown("---")

        # Multi-match insights button — requires all selected matches to have evidence + candidates
        all_ready = all(
            st.session_state["matches"][n]["evidence_packet"] is not None
            and st.session_state["matches"][n]["candidates"] is not None
            for n in sel
        )
        focus_player = st.session_state.get("multi_focus_player")
        no_focus = focus_player is None
        btn_disabled = not all_ready or no_focus

        if focus_player:
            st.info(f"🎯 Multi-match insights will be generated for **{focus_player['name']}**")

        if st.button(
            "🔀 Generate Multi-Match Insights (LLM)",
            disabled=btn_disabled,
            type="primary",
            use_container_width=False,
        ):
            try:
                matches_payload = []
                for idx, name in enumerate(sel):
                    mm = st.session_state["matches"][name]
                    matches_payload.append({
                        "label": _match_label(idx),
                        "match_metadata": mm["evidence_packet"]["match"],
                        "evidence_packet": mm["evidence_packet"],
                        "candidates": mm["candidates"],
                    })
                st.session_state["multi_insights"] = generate_multi_match_insights(
                    matches_payload,
                    focus_player=focus_player,
                    model=model,
                )
                st.session_state["multi_chat"] = []
                st.success(f"Multi-match insights generated for {focus_player['name']}.")
            except Exception as e:
                st.error(f"LLM failed: {e}")

        if not all_ready:
            st.caption("⚠️ Compute Evidence Packet + Candidate Points for all selected matches first.")
        if no_focus:
            st.caption("⚠️ Select a focus player in Section 2 first.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.markdown("---")
tabs = st.tabs(["Stats", "Insights", "Chat", "Export"])

# ---- Stats tab ----
with tabs[0]:
    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)
        if m and m["evidence_packet"] is not None:
            st.json(m["evidence_packet"], expanded=False)
        else:
            st.info("Compute Evidence Packet first (Section 3).")
    else:
        sel = st.session_state["selected_matches"]
        any_ep = False
        for name in sel:
            m = st.session_state["matches"].get(name)
            if m and m["evidence_packet"] is not None:
                any_ep = True
                st.markdown(f"#### {name}")
                st.json(m["evidence_packet"], expanded=False)
        if not any_ep:
            st.info("Compute Evidence Packets for selected matches first.")

# ---- Insights tab ----
with tabs[1]:
    def _render_insights(ins: dict, candidates: list | None, scope_label: str = ""):
        idx_to_point = {p["point_idx"]: p for p in (candidates or [])}

        st.markdown("### Entry summary")
        for line in ins.get("entry_summary", []):
            st.write(f"- {line}")

        st.markdown("### Insight Objects")
        insights_list = ins.get("insights", [])
        # Sort by priority desc
        insights_list = sorted(insights_list, key=lambda x: x.get("priority", 0), reverse=True)
        for i, obj in enumerate(insights_list, start=1):
            scope = obj.get("match_scope", "")
            scope_str = f" · scope: {scope}" if scope else ""
            label = f"{i}. {obj['title']} (priority {obj['priority']:.2f}){scope_str}"
            with st.expander(label, expanded=(i <= 3)):
                st.write(obj["summary"])
                st.markdown(f"**Coaching tip:** {obj['coaching_tip']}")
                st.markdown("**Evidence refs:** " + ", ".join(obj["evidence_refs"]))
                sp = obj.get("supporting_points", [])
                if sp:
                    st.markdown("**Supporting points:** " + ", ".join(map(str, sp)))
                    st.markdown("**Timestamp evidence:**")
                    for pid in sp[:8]:
                        p = idx_to_point.get(pid)
                        if p:
                            st.write(f"- #{pid} [{fmt_mmss(p['start_s'])}–{fmt_mmss(p['end_s'])}] — {p['rally_summary']}")

    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)
        if m and m["insights"] is not None:
            _render_insights(m["insights"], m["candidates"])
        else:
            st.info("Generate Insight Objects first (Section 3).")
    else:
        sel = st.session_state["selected_matches"]
        multi_ins = st.session_state["multi_insights"]

        if multi_ins is not None:
            # Combine candidates from all selected matches for timestamp lookup
            all_candidates: list = []
            for name in sel:
                mm = st.session_state["matches"].get(name)
                if mm and mm["candidates"]:
                    all_candidates.extend(mm["candidates"])
            _render_insights(multi_ins, all_candidates)
        else:
            st.info("Generate Multi-Match Insights first (Section 3).")

        # Optionally show per-match insights
        with st.expander("Per-match insights (individual)"):
            any_single = False
            for name in sel:
                mm = st.session_state["matches"].get(name)
                if mm and mm["insights"] is not None:
                    any_single = True
                    st.markdown(f"#### {name}")
                    _render_insights(mm["insights"], mm["candidates"])
            if not any_single:
                st.info("No per-match insights generated yet.")

# ---- Chat tab ----
with tabs[2]:
    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)

        if m is None or m["evidence_packet"] is None or m["insights"] is None or m["candidates"] is None:
            st.info("Compute Evidence + Candidates + Insights first (Section 3).")
        else:
            chat_history = st.session_state["chat"].get(name, [])

            for msg in chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            q = st.chat_input("Ask: Why did I lose? How was my 2nd serve? What should I work on?")
            if q:
                chat_history.append({"role": "user", "content": q})
                st.session_state["chat"][name] = chat_history
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        ans = answer_question(
                            q,
                            m["evidence_packet"]["match"],
                            m["evidence_packet"],
                            m["insights"],
                            m["candidates"],
                            model=model,
                        )
                        st.markdown(ans)
                        chat_history.append({"role": "assistant", "content": ans})
                        st.session_state["chat"][name] = chat_history

    else:  # multi
        sel = st.session_state["selected_matches"]
        multi_ins = st.session_state["multi_insights"]

        if len(sel) < 2:
            st.info("Select at least 2 matches in Section 2.")
        elif multi_ins is None:
            st.info("Generate Multi-Match Insights first (Section 3).")
        else:
            # Check all selected matches have candidates + evidence
            missing = [n for n in sel if not (
                st.session_state["matches"][n]["evidence_packet"] is not None
                and st.session_state["matches"][n]["candidates"] is not None
            )]
            if missing:
                st.warning(f"Missing evidence/candidates for: {', '.join(missing)}")
            else:
                for msg in st.session_state["multi_chat"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                focus_player = st.session_state.get("multi_focus_player")
                if focus_player:
                    st.caption(f"🎯 Chatting about **{focus_player['name']}** across {len(sel)} matches")

                q = st.chat_input("Ask about trends across all selected matches…")
                if q:
                    st.session_state["multi_chat"].append({"role": "user", "content": q})
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking…"):
                            matches_payload = []
                            for idx, name in enumerate(sel):
                                mm = st.session_state["matches"][name]
                                matches_payload.append({
                                    "label": _match_label(idx),
                                    "match_metadata": mm["evidence_packet"]["match"],
                                    "evidence_packet": mm["evidence_packet"],
                                    "insights": mm.get("insights"),
                                    "candidates": mm["candidates"],
                                })
                            ans = answer_question_multi(
                                q,
                                matches_payload,
                                multi_ins,
                                focus_player=focus_player,
                                model=model,
                            )
                            st.markdown(ans)
                            st.session_state["multi_chat"].append({"role": "assistant", "content": ans})

# ---- Export tab ----
with tabs[3]:
    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)
        if m:
            if m["evidence_packet"] is not None:
                st.download_button(
                    "Download evidence_packet.json",
                    json.dumps(m["evidence_packet"], indent=2, ensure_ascii=False),
                    f"{name}_evidence_packet.json",
                    "application/json",
                )
            if m["candidates"] is not None:
                st.download_button(
                    "Download candidate_points.json",
                    json.dumps(m["candidates"], indent=2, ensure_ascii=False),
                    f"{name}_candidate_points.json",
                    "application/json",
                )
            if m["insights"] is not None:
                st.download_button(
                    "Download insight_objects.json",
                    json.dumps(m["insights"], indent=2, ensure_ascii=False),
                    f"{name}_insight_objects.json",
                    "application/json",
                )
    else:
        sel = st.session_state["selected_matches"]
        for name in sel:
            m = st.session_state["matches"].get(name)
            if not m:
                continue
            st.markdown(f"#### {name}")
            c1, c2, c3 = st.columns(3)
            with c1:
                if m["evidence_packet"] is not None:
                    st.download_button(
                        "Evidence Packet",
                        json.dumps(m["evidence_packet"], indent=2, ensure_ascii=False),
                        f"{name}_evidence_packet.json",
                        "application/json",
                        key=f"dl_ep_{name}",
                    )
            with c2:
                if m["candidates"] is not None:
                    st.download_button(
                        "Candidate Points",
                        json.dumps(m["candidates"], indent=2, ensure_ascii=False),
                        f"{name}_candidate_points.json",
                        "application/json",
                        key=f"dl_cp_{name}",
                    )
            with c3:
                if m["insights"] is not None:
                    st.download_button(
                        "Single-Match Insights",
                        json.dumps(m["insights"], indent=2, ensure_ascii=False),
                        f"{name}_insight_objects.json",
                        "application/json",
                        key=f"dl_si_{name}",
                    )

        if st.session_state["multi_insights"] is not None:
            st.markdown("#### Multi-Match Insights")
            st.download_button(
                "Download multi_match_insights.json",
                json.dumps(st.session_state["multi_insights"], indent=2, ensure_ascii=False),
                "multi_match_insights.json",
                "application/json",
            )
