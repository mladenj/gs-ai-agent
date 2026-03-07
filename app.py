from __future__ import annotations
import sys
import time
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
    _default_insight_model,
    _default_chat_model,
)
from gs_match_ai.utils import fmt_mmss, sha256_file

DEFAULT_SEED = 7

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
st.title("🎾 GameSmart — Match AI Demo")
st.caption("Upload one or more L2 CSVs, generate single-match or multi-match coaching insights, and chat with the AI coach.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
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

def _cache_paths(csv_sha: str):
    cache_dir = ROOT / "data" / "processed" / csv_sha
    cache_dir.mkdir(parents=True, exist_ok=True)
    ep_path = cache_dir / "evidence_packet.json"
    cp_path = cache_dir / "candidate_points.json"
    ins_path = cache_dir / "insight_objects.json"
    return cache_dir, ep_path, cp_path, ins_path

def _save_json_atomic(path: Path, obj: dict | list):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def _load_json_safe(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        try:
            path.unlink()
        except Exception:
            pass
        return None

def _player_names_for_match(name: str) -> str:
    m = st.session_state["matches"].get(name)
    if not m or m.get("df") is None or len(m["df"]) == 0:
        return "—"
    row = m["df"].iloc[0]
    return f"{str(row['p1_fullname']).strip()} vs {str(row['p2_fullname']).strip()}"

def _match_status_summary(name: str) -> tuple[str, str]:
    m = st.session_state["matches"].get(name)
    if not m:
        return "—", "—"
    points = len(m["df"]) if m.get("df") is not None else 0
    cand_count = len(m["candidates"]) if m.get("candidates") else 0
    ins_count = len(m["insights"].get("insights", [])) if m.get("insights") else 0

    if m.get("insights") is not None:
        return "Analysis ready", f"{cand_count} key points · {ins_count} insights"
    if m.get("candidates") is not None or m.get("evidence_packet") is not None:
        return "Partially prepared", f"{cand_count} key points selected"
    return "Not analyzed", f"{points} points loaded"

def _priority_badge(priority: float) -> str:
    if priority >= 0.85:
        return "★★★★★ Main insight"
    if priority >= 0.70:
        return "★★★★ Strong signal"
    if priority >= 0.55:
        return "★★★ Useful pattern"
    if priority >= 0.40:
        return "★★ Supporting note"
    return "★ Minor observation"

def _analysis_sidebar_summary():
    matches = st.session_state["matches"]
    mode = st.session_state["analysis_mode"]
    selected_match = st.session_state.get("selected_match")
    selected_matches = st.session_state.get("selected_matches", [])

    st.markdown("### Analysis Summary")
    st.caption(f"Mode: {'Single Match' if mode == 'single' else 'Multi-Match'}")
    st.caption(f"Loaded matches: {len(matches)}")
    st.caption(f"Insight model: {_default_insight_model()}")
    st.caption(f"Chat model: {_default_chat_model()}")

    if mode == "single" and selected_match and selected_match in matches:
        m = matches[selected_match]
        st.markdown("---")
        st.caption("Current match")
        st.write(_player_names_for_match(selected_match))
        st.caption(selected_match)
        if m.get("candidates"):
            st.caption(f"Key points used: {len(m['candidates'])}")
        if m.get("insights"):
            st.caption(f"Insights ready: {len(m['insights'].get('insights', []))}")
    elif mode == "multi":
        st.markdown("---")
        st.caption("Current comparison")
        st.caption(f"Selected matches: {len(selected_matches)}")
        focus_player = st.session_state.get("multi_focus_player")
        if focus_player:
            st.caption(f"Focus player: {focus_player['name']}")
        if st.session_state.get("multi_insights"):
            st.caption(f"Joint insights ready: {len(st.session_state['multi_insights'].get('insights', []))}")

def _run_single_match_pipeline(name: str, m: dict, status=None, regenerate_insights: bool = False):
    cache_dir, ep_path, cp_path, ins_path = _cache_paths(m["csv_sha"])

    if not regenerate_insights:
        if m["evidence_packet"] is None:
            cached_ep = _load_json_safe(ep_path)
            if cached_ep is not None:
                m["evidence_packet"] = cached_ep
                if status:
                    status.write("✅ Match evidence loaded from cache")
            else:
                t0 = time.perf_counter()
                stats = compute_match_stats(m["df"])
                match_meta = {
                    "csv_name": name,
                    "csv_sha256": m["csv_sha"],
                    "players": stats["players"],
                }
                m["evidence_packet"] = {"match": match_meta, **stats}
                _save_json_atomic(ep_path, m["evidence_packet"])
                if status:
                    status.write(f"✅ Match evidence prepared in {time.perf_counter() - t0:.1f}s")

        if m["candidates"] is None:
            cached_cp = _load_json_safe(cp_path)
            if cached_cp is not None:
                m["candidates"] = cached_cp
                if status:
                    status.write(f"✅ Key points loaded: {len(m['candidates'])}")
            else:
                t1 = time.perf_counter()
                m["candidates"] = select_candidate_points(m["df"], max_points=None, seed=DEFAULT_SEED)
                _save_json_atomic(cp_path, m["candidates"])
                if status:
                    status.write(f"✅ Key points selected: {len(m['candidates'])} ({time.perf_counter() - t1:.1f}s)")

    if regenerate_insights:
        if ins_path.exists():
            try:
                ins_path.unlink()
            except Exception:
                pass
        m["insights"] = None

    if m["insights"] is None:
        cached_ins = _load_json_safe(ins_path)
        if cached_ins is not None:
            m["insights"] = cached_ins
            if status:
                status.write(f"✅ Coaching insights ready: {len(m['insights'].get('insights', []))} (cached)")
        else:
            if m["evidence_packet"] is None:
                cached_ep = _load_json_safe(ep_path)
                if cached_ep is None:
                    stats = compute_match_stats(m["df"])
                    match_meta = {"csv_name": name, "csv_sha256": m["csv_sha"], "players": stats["players"]}
                    m["evidence_packet"] = {"match": match_meta, **stats}
                    _save_json_atomic(ep_path, m["evidence_packet"])
                else:
                    m["evidence_packet"] = cached_ep

            if m["candidates"] is None:
                cached_cp = _load_json_safe(cp_path)
                if cached_cp is None:
                    m["candidates"] = select_candidate_points(m["df"], max_points=None, seed=DEFAULT_SEED)
                    _save_json_atomic(cp_path, m["candidates"])
                else:
                    m["candidates"] = cached_cp

            t2 = time.perf_counter()
            m["insights"] = generate_insights(
                m["evidence_packet"]["match"],
                m["evidence_packet"],
                m["candidates"],
                model=None,
            )
            _save_json_atomic(ins_path, m["insights"])
            if status:
                status.write(f"✅ Coaching insights generated: {len(m['insights'].get('insights', []))} ({time.perf_counter() - t2:.1f}s)")

    return cache_dir

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    _analysis_sidebar_summary()

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

    sample_dirs = [ROOT / "data" / "raw", ROOT / "data" / "local_raw"]
    samples = []
    for d in sample_dirs:
        if d.exists():
            samples.extend([str(p) for p in d.glob("*.csv")])
    samples = sorted(set(samples))
    sample_choice = st.selectbox("...or add a sample file", ["(none)"] + samples)

    if st.button("Load / Refresh CSVs", type="primary"):
        tmp_dir = ROOT / "data" / "processed"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        sources: list[tuple[str, Path]] = []

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
                    st.success(f"✅ Loaded '{name}'")
                except Exception as e:
                    st.error(f"Failed to load '{name}': {e}")

    if st.session_state["matches"]:
        st.markdown("### Loaded matches")
        rows = []
        for name, m in st.session_state["matches"].items():
            status, summary = _match_status_summary(name)
            rows.append({
                "Match file": name,
                "Players": _player_names_for_match(name),
                "Points": len(m["df"]) if m["df"] is not None else "—",
                "Status": status,
                "Summary": summary,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        remove_choice = st.selectbox(
            "Remove a match from session",
            ["(keep all)"] + list(st.session_state["matches"].keys()),
        )
        if st.button("Remove selected match") and remove_choice != "(keep all)":
            st.session_state["matches"].pop(remove_choice, None)
            st.session_state["chat"].pop(remove_choice, None)
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
# Section 3: Run Analysis
# ---------------------------------------------------------------------------
st.markdown("## 3 · Run Analysis")

mode = st.session_state["analysis_mode"]

if mode == "single":
    target_name = st.session_state["selected_match"]
    if not target_name or target_name not in st.session_state["matches"]:
        st.info("Select a match above.")
    else:
        m = st.session_state["matches"][target_name]
        cache_dir, ep_path, cp_path, ins_path = _cache_paths(m["csv_sha"])
        regen_disabled = not (
            (m["evidence_packet"] is not None or ep_path.exists())
            and (m["candidates"] is not None or cp_path.exists())
        )

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            run_btn = st.button("▶ Generate AI Analysis", type="primary")
        with btn_col2:
            regen_btn = st.button("🔄 Regenerate insights", disabled=regen_disabled)

        if run_btn:
            try:
                with st.status("Preparing match analysis…", expanded=True) as status:
                    status.update(label="Building match evidence and selecting key moments…", state="running")
                    _run_single_match_pipeline(target_name, m, status=status, regenerate_insights=False)
                    status.update(label="Analysis ready ✅", state="complete")

                N = len(m["insights"].get("insights", [])) if m["insights"] else 0
                st.success(
                    f"Analysis ready — {len(m['candidates'])} key points reviewed · {N} coaching insights generated · Chat is now available."
                )
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

        if regen_btn:
            try:
                with st.status("Refreshing coaching insights…", expanded=True) as status:
                    status.update(label="Regenerating insight objects…", state="running")
                    _run_single_match_pipeline(target_name, m, status=status, regenerate_insights=True)
                    status.update(label="Analysis refreshed ✅", state="complete")

                N = len(m["insights"].get("insights", [])) if m["insights"] else 0
                st.success(f"Insights refreshed — {N} coaching insights ready.")
            except Exception as e:
                st.error(f"Insight regeneration failed: {e}")

        with st.expander("Raw point preview (optional)", expanded=False):
            st.dataframe(
                m["df"][["point_idx","start_s","end_s","server","pt_won_by","end_type","rally_length","rally_summary"]].head(10),
                use_container_width=True,
            )

else:
    sel = st.session_state["selected_matches"]
    if len(sel) < 2:
        st.info("Select at least 2 matches above.")
    else:
        focus_player = st.session_state.get("multi_focus_player")
        if focus_player:
            st.info(f"🎯 Cross-match analysis will focus on **{focus_player['name']}**")

        cA, cB = st.columns([2, 1])
        with cA:
            run_all_btn = st.button("▶ Generate analysis for selected matches", type="primary")
        with cB:
            regen_all_btn = st.button("🔄 Regenerate match insights")

        if run_all_btn:
            try:
                with st.status("Preparing selected matches…", expanded=True) as status:
                    status.update(label="Running single-match analysis for each selected match…", state="running")
                    for name in sel:
                        mm = st.session_state["matches"][name]
                        status.write(f"**{name}**")
                        _run_single_match_pipeline(name, mm, status=status, regenerate_insights=False)
                        n_ins = len(mm["insights"].get("insights", [])) if mm["insights"] else 0
                        status.write(f"✅ {len(mm['candidates'])} key points · {n_ins} coaching insights")
                    status.update(label="Match analyses ready ✅", state="complete")
                st.success("Selected matches are ready for cross-match analysis.")
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

        if regen_all_btn:
            try:
                with st.status("Refreshing selected match insights…", expanded=True) as status:
                    status.update(label="Regenerating single-match insights…", state="running")
                    for name in sel:
                        mm = st.session_state["matches"][name]
                        status.write(f"**{name}**")
                        _run_single_match_pipeline(name, mm, status=status, regenerate_insights=True)
                        n_ins = len(mm["insights"].get("insights", [])) if mm["insights"] else 0
                        status.write(f"✅ {n_ins} insights refreshed")
                    status.update(label="Insights refreshed ✅", state="complete")
                st.success("Single-match insights refreshed for selected matches.")
            except Exception as e:
                st.error(f"Regeneration failed: {e}")

        st.markdown("---")

        all_ready = all(
            st.session_state["matches"][n]["evidence_packet"] is not None
            and st.session_state["matches"][n]["candidates"] is not None
            for n in sel
        )
        no_focus = focus_player is None
        btn_disabled = not all_ready or no_focus

        if st.button(
            "🔀 Generate Joint Multi-Match Insights",
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
                    model=None,
                )
                st.session_state["multi_chat"] = []
                st.success(f"Joint insights generated for {focus_player['name']}.")
            except Exception as e:
                st.error(f"LLM failed: {e}")

        if not all_ready:
            st.caption("⚠️ Run the match analysis for all selected matches first.")
        if no_focus:
            st.caption("⚠️ Select a focus player in Section 2 first.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
st.markdown("---")
tabs = st.tabs(["Stats", "Insights", "Chat", "Export"])

# ---- Stats tab ----
with tabs[0]:
    st.caption("This tab shows the computed evidence used by the AI, including platform-visible stats and deeper AI-only coaching metrics.")
    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)
        if m and m["evidence_packet"] is not None:
            st.json(m["evidence_packet"], expanded=False)
        else:
            st.info("Run analysis first.")
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
            st.info("Run analysis for selected matches first.")

# ---- Insights tab ----
with tabs[1]:
    def _render_insights(ins: dict, candidates: list | None, expand_top_n: int = 3):
        idx_to_point = {p["point_idx"]: p for p in (candidates or [])}

        entry_summary = ins.get("entry_summary", [])
        if entry_summary:
            st.markdown("### Summary")
            for line in entry_summary:
                st.write(f"- {line}")

        st.markdown("### Insights")
        insights_list = ins.get("insights", [])
        insights_list = sorted(insights_list, key=lambda x: x.get("priority", 0), reverse=True)

        for i, obj in enumerate(insights_list, start=1):
            badge = _priority_badge(float(obj.get("priority", 0)))
            label = f"{i}. {obj['title']} — {badge}"
            with st.expander(label, expanded=(i <= expand_top_n)):
                st.write(obj["summary"])
                st.markdown(f"**Coaching tip:** {obj['coaching_tip']}")
                sp = obj.get("supporting_points", [])
                if sp:
                    st.markdown("**Examples from the match:**")
                    for pid in sp[:8]:
                        p = idx_to_point.get(pid)
                        if p:
                            st.write(f"- #{pid} [{fmt_mmss(p['start_s'])}–{fmt_mmss(p['end_s'])}] — {p['rally_summary']}")
                if obj.get("evidence_refs"):
                    st.markdown("**Evidence:** " + ", ".join(obj["evidence_refs"]))

    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)
        if m and m["insights"] is not None:
            _render_insights(m["insights"], m["candidates"], expand_top_n=3)
        else:
            st.info("Generate AI analysis first.")
    else:
        sel = st.session_state["selected_matches"]
        multi_ins = st.session_state["multi_insights"]

        st.markdown("## Joint Insights")
        st.caption("Patterns and trends across the selected matches.")
        if multi_ins is not None:
            all_candidates: list = []
            for name in sel:
                mm = st.session_state["matches"].get(name)
                if mm and mm["candidates"]:
                    all_candidates.extend(mm["candidates"])
            _render_insights(multi_ins, all_candidates, expand_top_n=3)
        else:
            st.info("Generate Joint Multi-Match Insights first.")

        st.divider()
        st.markdown("## Individual Match Insights")
        st.caption("Single-match coaching insights, shown separately for each selected match.")

        any_single = False
        for name in sel:
            mm = st.session_state["matches"].get(name)
            if mm and mm["insights"] is not None:
                any_single = True
                st.markdown(f"### {name}")
                st.caption(_player_names_for_match(name))
                _render_insights(mm["insights"], mm["candidates"], expand_top_n=2)
                st.divider()
        if not any_single:
            st.info("No single-match insights generated yet.")

# ---- Chat tab ----
with tabs[2]:
    if mode == "single":
        name = st.session_state["selected_match"]
        m = st.session_state["matches"].get(name)

        if m is None or m["evidence_packet"] is None or m["insights"] is None or m["candidates"] is None:
            st.info("Generate AI analysis first.")
        else:
            st.caption("Ask about this match’s patterns, pressure moments, serve/return performance, or training priorities.")
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
                            model=None,
                        )
                        st.markdown(ans)
                        chat_history.append({"role": "assistant", "content": ans})
                        st.session_state["chat"][name] = chat_history

    else:
        sel = st.session_state["selected_matches"]
        multi_ins = st.session_state["multi_insights"]

        if len(sel) < 2:
            st.info("Select at least 2 matches in Analysis Mode.")
        elif multi_ins is None:
            st.info("Generate Joint Multi-Match Insights first.")
        else:
            missing = [n for n in sel if not (
                st.session_state["matches"][n]["evidence_packet"] is not None
                and st.session_state["matches"][n]["candidates"] is not None
            )]
            if missing:
                st.warning(f"Missing analysis for: {', '.join(missing)}")
            else:
                focus_player = st.session_state.get("multi_focus_player")
                if focus_player:
                    st.caption(f"Ask about recurring strengths, weaknesses, changes across matches, or training priorities for **{focus_player['name']}**.")

                for msg in st.session_state["multi_chat"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

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
                                model=None,
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
