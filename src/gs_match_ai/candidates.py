from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def select_candidate_points(df: pd.DataFrame, max_points: int = 60, seed: int = 7) -> List[Dict[str, Any]]:
    use = df[df["discard_point"] != True].copy()  # noqa: E712
    cand = set()

    def add_points(sub: pd.DataFrame, limit: int | None = None):
        if limit is None:
            for i in sub.index:
                cand.add(int(i))
        else:
            for i in list(sub.index)[:limit]:
                cand.add(int(i))

    for col in ["breakpoint", "setpoint", "matchpoint", "gamepoint"]:
        if col in use.columns:
            add_points(use[use[col].notna()])

    for col in ["double_fault", "winner", "error"]:
        if col in use.columns:
            add_points(use[use[col].notna()], limit=20)

    if "rally_length" in use.columns:
        add_points(use.sort_values("rally_length", ascending=False).head(10))

    rng = np.random.default_rng(seed)
    remaining = [i for i in use.index if int(i) not in cand]
    if remaining:
        k = min(10, len(remaining))
        add_points(use.loc[rng.choice(remaining, size=k, replace=False)])

    ordered = sorted(cand)[:max_points]

    out = []
    for i in ordered:
        r = use.loc[i]
        out.append({
            "point_idx": int(r["point_idx"]),
            "start_s": float(r["start_s"]),
            "end_s": float(r["end_s"]),
            "server": str(r["server"]),
            "pt_won_by": str(r["pt_won_by"]),
            "end_type": str(r.get("end_type", "OTHER")),
            "rally_length": None if pd.isna(r.get("rally_length")) else int(r.get("rally_length")),
            "rally_summary": "" if pd.isna(r.get("rally_summary")) else str(r.get("rally_summary")),
            "rally_desc": "" if pd.isna(r.get("rally_desc")) else str(r.get("rally_desc")),
        })
    return out
