from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_output(use: pd.DataFrame, ordered_indices: list[int]) -> List[Dict[str, Any]]:
    """Convert a sorted list of DataFrame indices to the standard candidate-point dicts."""
    out = []
    for i in ordered_indices:
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


# ---------------------------------------------------------------------------
# Auto selection (quality-first)
# ---------------------------------------------------------------------------

@st.cache_data
def select_candidate_points_auto(df: pd.DataFrame, seed: int = 7) -> List[Dict[str, Any]]:
    """
    Automatically choose the best K candidate points from *df* using a
    quality-first, deterministic strategy.

    K = clamp(round(0.35 * total_valid_points), 45, 90)

    Selection layers (deduplication applied throughout):
      1. Must-include: any break/game/set/match/pressure point, double_fault,
         or highlights == True
      2. Top-10 longest rallies
      3. Top-10 biggest momentum swings (|Δ(p1_momentum - p2_momentum)|)
      4. Balanced fill across buckets: rally length, serve number, end type,
         serve direction
      5. Deterministic random sample for any remaining slots
    """
    use = df[df["discard_point"] != True].copy()  # noqa: E712
    total = len(use)
    K = int(max(45, min(90, round(0.35 * total))))

    cand: set[int] = set()

    def _add(sub: pd.DataFrame, limit: int | None = None) -> None:
        indices = list(sub.index) if limit is None else list(sub.index)[:limit]
        for i in indices:
            cand.add(int(i))

    # ------------------------------------------------------------------
    # Layer 1 – must-include high-importance points
    # ------------------------------------------------------------------
    for col in ("breakpoint", "gamepoint", "setpoint", "matchpoint", "pressure_point"):
        if col in use.columns:
            _add(use[use[col].notna()])

    if "double_fault" in use.columns:
        _add(use[use["double_fault"].notna()])

    if "highlights" in use.columns:
        _add(use[use["highlights"] == True])  # noqa: E712

    # ------------------------------------------------------------------
    # Layer 2 – top 10 long rallies
    # ------------------------------------------------------------------
    if "rally_length" in use.columns:
        _add(use.sort_values("rally_length", ascending=False).head(10))

    # ------------------------------------------------------------------
    # Layer 3 – top 10 momentum swings
    # ------------------------------------------------------------------
    if "p1_momentum" in use.columns and "p2_momentum" in use.columns:
        mom_df = use[["p1_momentum", "p2_momentum"]].copy().dropna()
        if len(mom_df) > 1:
            delta = (mom_df["p1_momentum"] - mom_df["p2_momentum"]).diff().abs()
            top_swing_idx = delta.nlargest(10).index.tolist()
            for i in top_swing_idx:
                cand.add(int(i))

    # ------------------------------------------------------------------
    # Layer 4 – balanced fill across coverage buckets
    # ------------------------------------------------------------------
    if len(cand) < K:
        buckets: list[pd.DataFrame] = []

        # Rally-length buckets
        if "rally_length" in use.columns:
            rl = use["rally_length"].fillna(0)
            buckets.append(use[rl.between(1, 4)])
            buckets.append(use[rl.between(5, 8)])
            buckets.append(use[rl >= 9])

        # Serve number buckets
        if "serve_num_used" in use.columns:
            for v in (1, 2):
                buckets.append(use[use["serve_num_used"] == v])

        # End-type buckets
        if "end_type" in use.columns:
            for v in ("WINNER", "ERROR", "DOUBLE_FAULT", "OTHER"):
                buckets.append(use[use["end_type"] == v])

        # Serve direction buckets
        if "final_serve_dir" in use.columns:
            for v in ("T", "BODY", "WIDE", "UNKNOWN"):
                buckets.append(use[use["final_serve_dir"] == v])

        # Round-robin: add one point per bucket until K reached
        any_added = True
        bucket_ptrs = [0] * len(buckets)
        while len(cand) < K and any_added:
            any_added = False
            for b_idx, bucket in enumerate(buckets):
                if len(cand) >= K:
                    break
                rows_not_in_cand = [i for i in bucket.index if int(i) not in cand]
                ptr = bucket_ptrs[b_idx]
                while ptr < len(rows_not_in_cand):
                    candidate_i = int(rows_not_in_cand[ptr])
                    ptr += 1
                    if candidate_i not in cand:
                        cand.add(candidate_i)
                        any_added = True
                        break
                bucket_ptrs[b_idx] = ptr

    # ------------------------------------------------------------------
    # Layer 5 – random fill for remaining slots
    # ------------------------------------------------------------------
    if len(cand) < K:
        rng = np.random.default_rng(seed)
        remaining = [int(i) for i in use.index if int(i) not in cand]
        if remaining:
            n_pick = min(K - len(cand), len(remaining))
            chosen = rng.choice(remaining, size=n_pick, replace=False)
            for i in chosen:
                cand.add(int(i))

    # ------------------------------------------------------------------
    # Build output sorted chronologically, capped at K
    # ------------------------------------------------------------------
    ordered = sorted(cand)[:K]
    return _build_output(use, ordered)


# ---------------------------------------------------------------------------
# Legacy / public API  (signature unchanged)
# ---------------------------------------------------------------------------

@st.cache_data
def select_candidate_points(
    df: pd.DataFrame,
    max_points: int | None = 60,
    seed: int = 7,
) -> List[Dict[str, Any]]:
    """
    Public API for candidate-point selection.

    • If *max_points* is None  → delegates to the quality-first auto algorithm.
    • If *max_points* is given → uses the original heuristic (legacy behaviour).

    The function signature is kept stable so any downstream callers continue to work.
    """
    if max_points is None:
        return select_candidate_points_auto(df, seed=seed)

    # ------------------------------------------------------------------
    # Legacy path (kept for backward-compatibility with explicit max_points)
    # ------------------------------------------------------------------
    use = df[df["discard_point"] != True].copy()  # noqa: E712
    cand: set[int] = set()

    def add_points(sub: pd.DataFrame, limit: int | None = None) -> None:
        if limit is None:
            for i in sub.index:
                cand.add(int(i))
        else:
            for i in list(sub.index)[:limit]:
                cand.add(int(i))

    for col in ("breakpoint", "setpoint", "matchpoint", "gamepoint"):
        if col in use.columns:
            add_points(use[use[col].notna()])

    for col in ("double_fault", "winner", "error"):
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
    return _build_output(use, ordered)
