from __future__ import annotations

"""
GameSmart AI Agent (local) — Match Stats Engine

This file is used by the **local Streamlit AI Agent demo** ONLY.
It does NOT modify the original CSV; it derives all needed metrics from the columns already present.

Compatibility contract (do not break):
- `compute_match_stats(df: pd.DataFrame) -> Dict[str, Any]`
- The return object MUST include:
  - `players`
  - `totals`
  - `kpis.serve` and `kpis.return` (to keep downstream prompts/UI stable)

This implementation intentionally separates:
1) `kpis.platform_visible`   -> Stats currently displayed on the GameSmart platform match page.
2) `kpis.llm_extended`       -> Extra "LLM-only" stats for better insight generation (not shown on platform for now).

All calculations are derived from the L2 CSV columns (and are robust if the caller passes either:
- the raw L2 CSV dataframe, or
- a normalized dataframe with helper columns like `serve_num_used`, `s1_fault`, `final_serve_dir`, `start_s`, `end_s`).

Best practices used:
- Keep metrics **auditable** (counts + denominators).
- Avoid hallucination: LLM should reference only these computed values.
- Keep computation deterministic and stable across runs.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import re
import numpy as np
import pandas as pd
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _DummyStreamlit:
        def cache_data(self, func=None, **kwargs):
            if func is None:
                return lambda f: f
            return func
    st = _DummyStreamlit()  # type: ignore


# ---------------------------
# Small helpers
# ---------------------------

def _pct(x: float) -> float:
    return float(np.round(x * 100.0, 2))

def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return n / d

def _count_eq(series: pd.Series, value: str) -> int:
    if series is None:
        return 0
    return int((series == value).sum())

def _to_bool_series(s: pd.Series) -> pd.Series:
    # Handles booleans already present and mixed/object columns.
    if s.dtype == bool:
        return s
    return s.fillna(False).astype(bool)

def _col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    return df[name] if name in df.columns else None

def _entropy_norm(probs: List[float]) -> Optional[float]:
    """
    Normalized entropy in [0,1] for a discrete distribution.
    0 = fully predictable; 1 = maximally mixed.
    """
    probs = [p for p in probs if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p, 2) for p in probs)
    h_max = math.log(len(probs), 2)
    return float(h / h_max) if h_max > 0 else 0.0


# ---------------------------
# Parsing helpers (derived from existing CSV text columns)
# ---------------------------

_FAULT_RE = re.compile(r"fault\s*\(([^)]+)\)", re.IGNORECASE)

def _parse_serve_dir(text: Any) -> str:
    """
    Parse serve direction from the L2 serve text.
    Expected phrases in CSV:
      - "down the T"
      - "to body"
      - "wide"
    """
    if pd.isna(text):
        return "UNKNOWN"
    raw = str(text).lower()
    if "down the t" in raw:
        return "T"
    if "to body" in raw:
        return "BODY"
    if "wide" in raw:
        return "WIDE"
    return "UNKNOWN"

def _is_fault(text: Any) -> bool:
    if pd.isna(text):
        return False
    return "fault" in str(text).lower()

def _fault_type(text: Any) -> Optional[str]:
    if pd.isna(text):
        return None
    m = _FAULT_RE.search(str(text))
    if not m:
        return None
    return m.group(1).strip().upper()


# ---------------------------
# Derived columns (do NOT write back to original CSV; only to local copy)
# ---------------------------

def _ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Filter out discarded points later, but keep this function generic.

    # start/end seconds
    if "start_s" not in out.columns:
        if "vid_second" in out.columns:
            out["start_s"] = pd.to_numeric(out["vid_second"], errors="coerce")
        else:
            out["start_s"] = np.nan
    if "end_s" not in out.columns:
        if "end" in out.columns:
            out["end_s"] = pd.to_numeric(out["end"], errors="coerce")
        else:
            out["end_s"] = np.nan

    # serve_num_used
    if "serve_num_used" not in out.columns:
        if "2nd_serve" in out.columns:
            out["serve_num_used"] = (~out["2nd_serve"].isna()).astype(int).replace({0: 1, 1: 2})
        else:
            out["serve_num_used"] = 1

    # s1_fault, s2_fault
    if "s1_fault" not in out.columns and "1st_serve" in out.columns:
        out["s1_fault"] = out["1st_serve"].apply(_is_fault)
    if "s2_fault" not in out.columns and "2nd_serve" in out.columns:
        out["s2_fault"] = out["2nd_serve"].apply(_is_fault)
    if "s1_fault" not in out.columns:
        out["s1_fault"] = False
    if "s2_fault" not in out.columns:
        out["s2_fault"] = False

    # final serve direction (direction of the serve that started the rally)
    if "final_serve_dir" not in out.columns:
        if "1st_serve" in out.columns:
            s1_dir = out["1st_serve"].apply(_parse_serve_dir)
        else:
            s1_dir = pd.Series(["UNKNOWN"] * len(out))
        if "2nd_serve" in out.columns:
            s2_dir = out["2nd_serve"].apply(_parse_serve_dir)
        else:
            s2_dir = pd.Series(["UNKNOWN"] * len(out))

        # If 2nd serve used, use 2nd direction, else 1st direction
        out["final_serve_dir"] = np.where(out["serve_num_used"] == 2, s2_dir, s1_dir)

    # point_idx (stable 1-based id)
    if "point_idx" not in out.columns:
        out = out.reset_index(drop=True)
        out["point_idx"] = out.index + 1

    return out


# ---------------------------
# Score-state tags from existing point score columns
# ---------------------------

def _score_state_tags(use: pd.DataFrame) -> pd.DataFrame:
    out = use.copy()
    if "p1_points" in out.columns and "p2_points" in out.columns:
        p1p = pd.to_numeric(out["p1_points"], errors="coerce").fillna(-1)
        p2p = pd.to_numeric(out["p2_points"], errors="coerce").fillna(-1)
        out["is_30_30"] = (p1p == 30) & (p2p == 30)
        out["is_deuce"] = (p1p == 40) & (p2p == 40)
        out["is_ad_point"] = ((p1p == 50) & (p2p == 40)) | ((p1p == 40) & (p2p == 50))
    else:
        out["is_30_30"] = False
        out["is_deuce"] = False
        out["is_ad_point"] = False

    # union pressure tag (LLM-only, broader than platform's pressure_point)
    pressure_cols = [c for c in ["pressure_point", "breakpoint", "gamepoint", "setpoint", "matchpoint"] if c in out.columns]
    any_pressure = False
    for c in pressure_cols:
        any_pressure = any_pressure | out[c].notna()
    out["is_pressure_union"] = any_pressure
    out["is_clutch_union"] = out["is_30_30"] | out["is_deuce"] | out["is_ad_point"] | out["is_pressure_union"]
    return out


# ---------------------------
# Game segmentation for flow metrics (uses existing p1_games/p2_games progression)
# ---------------------------

def _extract_games(use: pd.DataFrame, p1_id: str, p2_id: str) -> List[Dict[str, Any]]:
    """
    Extract a game-level sequence from point-level data.

    Uses the fact that p1_games/p2_games reset only at set boundary and
    increment when a game is won. We detect game boundaries by changes
    in (p1_games, p2_games) compared to previous point row.

    Returns a list of games with:
      - game_index (1-based)
      - server_id
      - winner_id
      - start_point_idx, end_point_idx
    """
    if "p1_games" not in use.columns or "p2_games" not in use.columns:
        return []

    g1 = pd.to_numeric(use["p1_games"], errors="coerce").fillna(method="ffill").fillna(0).astype(int).values
    g2 = pd.to_numeric(use["p2_games"], errors="coerce").fillna(method="ffill").fillna(0).astype(int).values
    server = use["server"].astype(str).values
    point_idx = use["point_idx"].astype(int).values

    games: List[Dict[str, Any]] = []
    current_start = 0
    current_server = server[0] if len(server) else None
    last_g1, last_g2 = g1[0], g2[0]

    for i in range(1, len(use)):
        if g1[i] != last_g1 or g2[i] != last_g2:
            # previous point ended a game
            winner_id = p1_id if (g1[i] > last_g1) else p2_id if (g2[i] > last_g2) else None
            games.append({
                "game_index": len(games) + 1,
                "server_id": current_server,
                "winner_id": winner_id,
                "start_point_idx": int(point_idx[current_start]),
                "end_point_idx": int(point_idx[i - 1]),
                "p1_games_after": int(g1[i]),
                "p2_games_after": int(g2[i]),
            })
            # new game starts at i
            current_start = i
            current_server = server[i]
            last_g1, last_g2 = g1[i], g2[i]

    # last game winner might not be detected if dataset ends mid-game
    # so we only include completed games above.
    return games


# ---------------------------
# Main entry: compute match stats
# ---------------------------

@st.cache_data
def compute_match_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns:
      {
        "players": {"p1": {...}, "p2": {...}},
        "totals": {...},
        "kpis": {
          "serve": {...},    # compatibility
          "return": {...},   # compatibility
          "platform_visible": {...}, # matches platform stats
          "llm_extended": {...},     # extra stats for LLM insights
        }
      }
    """
    # NOTE: caller may pass raw df; derive helper cols here.
    base = _ensure_derived_columns(df)

    # Remove discarded points for all computations (keep original rows intact elsewhere)
    use = base[base["discard_point"] != True].copy()  # noqa: E712
    if len(use) == 0:
        return {"players": {}, "totals": {}, "kpis": {}}

    # Identify players
    p1_id = str(use.iloc[0]["p1_id"])
    p2_id = str(use.iloc[0]["p2_id"])
    p1_name = str(use.iloc[0]["p1_fullname"]).strip()
    p2_name = str(use.iloc[0]["p2_fullname"]).strip()

    # Score-state tags
    use = _score_state_tags(use)

    # Convenience mappings
    pid_to_key = {p1_id: "p1", p2_id: "p2"}

    # ---------------------------
    # Core totals
    # ---------------------------
    total_points = int(len(use))
    p1_points_won = int((use["pt_won_by"] == p1_id).sum())
    p2_points_won = int((use["pt_won_by"] == p2_id).sum())

    # Active/removed time (platform)
    active_time_s = int(pd.to_numeric(use.get("point_duration"), errors="coerce").fillna(0).sum()) if "point_duration" in use.columns else None
    span_s = None
    removed_time_s = None
    if "start_s" in use.columns and "end_s" in use.columns:
        st0 = float(pd.to_numeric(use["start_s"], errors="coerce").min())
        en0 = float(pd.to_numeric(use["end_s"], errors="coerce").max())
        if not (math.isnan(st0) or math.isnan(en0)) and en0 >= st0:
            span_s = int(round(en0 - st0))
            if active_time_s is not None:
                removed_time_s = max(0, span_s - active_time_s)

    # ---------------------------
    # Platform-visible blocks
    # ---------------------------

    def _format_mmss(total_seconds: Optional[int]) -> Optional[str]:
        if total_seconds is None:
            return None
        s = int(total_seconds)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m}m{s:02d}s"

    def _platform_points_won() -> Dict[str, Any]:
        return {
            "points": {
                "total": total_points,
                "won": {"p1": p1_points_won, "p2": p2_points_won},
                "won_pct": {
                    "p1": _pct(p1_points_won / total_points) if total_points else 0.0,
                    "p2": _pct(p2_points_won / total_points) if total_points else 0.0,
                },
            }
        }

    def _platform_first_serve_in_pct(pid: str) -> Dict[str, Any]:
        pts = use[use["server"] == pid]
        total = len(pts)
        if total == 0:
            return {"in_pct": None, "in": 0, "attempts": 0}
        s1_in = (~pts["s1_fault"]).sum()
        return {"in_pct": _pct(s1_in / total), "in": int(s1_in), "attempts": int(total)}

    def _platform_aces(pid: str) -> int:
        # ace column may be player id, boolean, or NaN
        if "ace" not in use.columns:
            return 0
        col = use["ace"]
        # If stored as player id
        if col.dtype == object:
            return int((col == pid).sum())
        # If boolean-ish numeric, count rows where ace is truthy and pt_won_by == server (best-effort)
        try:
            truthy = pd.to_numeric(col, errors="coerce").fillna(0) > 0
            # If truthy, attribute to server of that point
            return int(((truthy) & (use["server"] == pid)).sum())
        except Exception:
            return 0

    def _platform_double_faults(pid: str) -> int:
        return int((use.get("double_fault") == pid).sum()) if "double_fault" in use.columns else 0

    def _platform_winners(pid: str) -> int:
        return int((use.get("winner") == pid).sum()) if "winner" in use.columns else 0

    def _platform_unforced_errors(pid: str) -> int:
        # In L2 we only have `error` (proxy for UE at this stage).
        return int((use.get("error") == pid).sum()) if "error" in use.columns else 0

    def _platform_breakpoints(pid: str) -> Dict[str, Any]:
        if "breakpoint" not in use.columns:
            return {"earned": 0, "converted": 0, "conversion_pct": None}
        bp = use[use["breakpoint"].notna()]
        earned = int((bp["breakpoint"] == pid).sum())
        converted = int(((bp["breakpoint"] == pid) & (bp["pt_won_by"] == pid)).sum())
        return {"earned": earned, "converted": converted, "conversion_pct": None if earned == 0 else _pct(converted / earned)}

    def _platform_pressure_points(pid: str) -> Dict[str, Any]:
        # Uses pressure_point column directly (platform uses this definition).
        if "pressure_point" not in use.columns:
            return {"played": 0, "won": 0, "won_pct": None}
        pp = use[use["pressure_point"].notna()]
        played = int((pp["pressure_point"] == pid).sum())
        won = int(((pp["pressure_point"] == pid) & (pp["pt_won_by"] == pid)).sum())
        return {"played": played, "won": won, "won_pct": None if played == 0 else _pct(won / played)}

    def _platform_net_points(pid: str) -> Dict[str, Any]:
        col = "p1_net_appearance" if pid == p1_id else "p2_net_appearance"
        if col not in use.columns:
            return {"played": 0, "won": 0, "won_pct": None}
        net = _to_bool_series(use[col])
        played = int(net.sum())
        won = int((net & (use["pt_won_by"] == pid)).sum())
        return {"played": played, "won": won, "won_pct": None if played == 0 else _pct(won / played)}

    def _platform_rally_analysis() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        buckets = [
            ("short", "rally_1-4", "1-4"),
            ("medium", "rally_5-8", "5-8"),
            ("long", "rally_9", "9+"),
        ]
        for key, col, label in buckets:
            if col not in use.columns:
                continue
            mask = _to_bool_series(use[col])
            pts = use[mask]
            total = int(len(pts))
            if total_points == 0:
                pct = 0.0
            else:
                pct = _pct(total / total_points)
            wins_p1 = int((pts["pt_won_by"] == p1_id).sum())
            wins_p2 = int((pts["pt_won_by"] == p2_id).sum())
            out[key] = {
                "label": label,
                "points": total,
                "points_pct": pct,
                "wins": {"p1": wins_p1, "p2": wins_p2},
            }
        return out

    def _platform_shot_breakdown() -> Dict[str, Any]:
        """
        Platform shows categories + (at least) FH winners/errors.
        We compute per-player counts for each category where we can infer them.
        """
        def _cnt(col: str, pid: str) -> int:
            return int((use.get(col) == pid).sum()) if col in use.columns else 0

        categories = {
            "forehand": ["forehand_winner", "forehand_error"],
            "backhand": ["backhand_winner", "backhand_error"],
            "overhead": ["smash"],
            "drop_shot": ["drop_shot"],
            "lob": ["lob"],
        }

        per_player = {"p1": {}, "p2": {}}
        totals = {}

        for cat, cols in categories.items():
            # total occurrences (any player)
            totals[cat] = int(sum(use[c].notna().sum() for c in cols if c in use.columns))
            for pid, key in [(p1_id, "p1"), (p2_id, "p2")]:
                # count if player id is stored in those cols
                per_player[key][cat] = int(sum((use[c] == pid).sum() for c in cols if c in use.columns))

        return {
            "categories": ["forehand", "backhand", "overhead", "drop_shot", "lob"],
            "counts_total": totals,
            "counts_by_player": per_player,
            "forehand_winners": {"p1": _cnt("forehand_winner", p1_id), "p2": _cnt("forehand_winner", p2_id)},
            "forehand_errors": {"p1": _cnt("forehand_error", p1_id), "p2": _cnt("forehand_error", p2_id)},
            "backhand_winners": {"p1": _cnt("backhand_winner", p1_id), "p2": _cnt("backhand_winner", p2_id)},
            "backhand_errors": {"p1": _cnt("backhand_error", p1_id), "p2": _cnt("backhand_error", p2_id)},
        }

    def _serve_breakdown(pid: str, subset_mask: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Count serves by direction and wins behind that direction.
        subset_mask can be used to restrict to 1st-serve points or 2nd-serve points.
        """
        pts = use[use["server"] == pid]
        if subset_mask is not None:
            pts = pts[subset_mask.loc[pts.index]]
        total = len(pts)
        if total == 0:
            return {"total": 0, "by_dir": {}}
        by_dir = {}
        for d in ["WIDE", "T", "BODY", "UNKNOWN"]:
            m = (pts["final_serve_dir"].fillna("UNKNOWN").astype(str) == d)
            count = int(m.sum())
            wins = int((m & (pts["pt_won_by"] == pid)).sum())
            if count > 0:
                by_dir[d] = {"serves": count, "wins": wins, "win_pct": _pct(wins / count) if count else None}
        return {"total": int(total), "by_dir": by_dir}

    def _platform_serve_breakdown() -> Dict[str, Any]:
        """
        Mirrors platform: All Serves / 1st Serve / 2nd Serve
        Using:
          - All: final_serve_dir on all service points
          - 1st: service points where serve_num_used == 1
          - 2nd: service points where serve_num_used == 2
        """
        out: Dict[str, Any] = {"all_serves": {}, "first_serve": {}, "second_serve": {}}
        for pid, key in [(p1_id, "p1"), (p2_id, "p2")]:
            out["all_serves"][key] = _serve_breakdown(pid)
            out["first_serve"][key] = _serve_breakdown(pid, subset_mask=(use["serve_num_used"] == 1))
            out["second_serve"][key] = _serve_breakdown(pid, subset_mask=(use["serve_num_used"] == 2))
        return out

    platform_visible = {
        # Timing/header
        "timing": {
            "active_time_s": active_time_s,
            "active_time_str": _format_mmss(active_time_s),
            "span_s": span_s,
            "removed_time_s": removed_time_s,
            "removed_time_str": _format_mmss(removed_time_s),
            "points": total_points,
        },
        # Momentum series (if present)
        "momentum_series": {
            "p1": use["p1_momentum"].dropna().astype(float).tolist() if "p1_momentum" in use.columns else [],
            "p2": use["p2_momentum"].dropna().astype(float).tolist() if "p2_momentum" in use.columns else [],
        },
        # Core match stats
        "points_won": _platform_points_won(),
        "first_serve_in_pct": {
            "p1": _platform_first_serve_in_pct(p1_id),
            "p2": _platform_first_serve_in_pct(p2_id),
        },
        "aces": {"p1": _platform_aces(p1_id), "p2": _platform_aces(p2_id)},
        "double_faults": {"p1": _platform_double_faults(p1_id), "p2": _platform_double_faults(p2_id)},
        "winners": {"p1": _platform_winners(p1_id), "p2": _platform_winners(p2_id)},
        "unforced_errors": {"p1": _platform_unforced_errors(p1_id), "p2": _platform_unforced_errors(p2_id)},
        "break_points": {"p1": _platform_breakpoints(p1_id), "p2": _platform_breakpoints(p2_id)},
        "pressure_points": {"p1": _platform_pressure_points(p1_id), "p2": _platform_pressure_points(p2_id)},
        "net_points": {"p1": _platform_net_points(p1_id), "p2": _platform_net_points(p2_id)},
        "rally_analysis": _platform_rally_analysis(),
        "shot_breakdown": _platform_shot_breakdown(),
        "serve_breakdown": _platform_serve_breakdown(),
    }

    # ---------------------------
    # Compatibility serve/return KPI blocks (used by existing app/prompts)
    # ---------------------------

    def serve_block(pid: str) -> Dict[str, Any]:
        pts = use[use["server"] == pid]
        total = len(pts)
        if total == 0:
            return {"service_points": 0}

        first_in = (~pts["s1_fault"]).mean()
        second_used = (pts["serve_num_used"] == 2).mean()
        srv_win = (pts["pt_won_by"] == pid).mean()

        first_pts = pts[~pts["s1_fault"]]
        second_pts = pts[pts["serve_num_used"] == 2]
        first_win = (first_pts["pt_won_by"] == pid).mean() if len(first_pts) else np.nan
        second_win = (second_pts["pt_won_by"] == pid).mean() if len(second_pts) else np.nan

        dir_counts = pts["final_serve_dir"].fillna("UNKNOWN").value_counts().to_dict()
        dir_win = (
            pts.groupby("final_serve_dir")
            .apply(lambda g: float((g["pt_won_by"] == pid).mean()))
            .to_dict()
        )

        df_count = _platform_double_faults(pid)

        return {
            "service_points": int(total),
            "first_serve_in_pct": _pct(first_in),
            "second_serve_used_pct": _pct(second_used),
            "service_points_won_pct": _pct(srv_win),
            "first_serve_points_won_pct": None if np.isnan(first_win) else _pct(first_win),
            "second_serve_points_won_pct": None if np.isnan(second_win) else _pct(second_win),
            "double_faults": df_count,
            "final_serve_dir_counts": {str(k): int(v) for k, v in dir_counts.items()},
            "final_serve_dir_win_pct": {str(k): _pct(v) for k, v in dir_win.items()},
        }

    def return_block(pid: str) -> Dict[str, Any]:
        pts = use[use["server"] != pid]
        total = len(pts)
        if total == 0:
            return {"return_points": 0}
        won = (pts["pt_won_by"] == pid).mean()

        opp_first = pts[~pts["s1_fault"]]
        opp_second = pts[pts["serve_num_used"] == 2]
        won_opp_first = (opp_first["pt_won_by"] == pid).mean() if len(opp_first) else np.nan
        won_opp_second = (opp_second["pt_won_by"] == pid).mean() if len(opp_second) else np.nan

        return {
            "return_points": int(total),
            "return_points_won_pct": _pct(won),
            "return_points_won_vs_opp_first_pct": None if np.isnan(won_opp_first) else _pct(won_opp_first),
            "return_points_won_vs_opp_second_pct": None if np.isnan(won_opp_second) else _pct(won_opp_second),
        }

    # ---------------------------
    # LLM-only extended stats (computed from available L2 columns)
    # ---------------------------

    def _context_mask(role: str, pid: str) -> pd.Series:
        if role == "serve":
            return (use["server"] == pid)
        if role == "return":
            return (use["server"] != pid)
        return pd.Series([True] * len(use), index=use.index)

    def _rally_bucket_masks() -> Dict[str, pd.Series]:
        masks = {}
        if "rally_1-4" in use.columns:
            masks["short"] = _to_bool_series(use["rally_1-4"])
        else:
            masks["short"] = (pd.to_numeric(use.get("rally_length"), errors="coerce") <= 4).fillna(False) if "rally_length" in use.columns else pd.Series([False]*len(use), index=use.index)
        if "rally_5-8" in use.columns:
            masks["medium"] = _to_bool_series(use["rally_5-8"])
        else:
            rl = pd.to_numeric(use.get("rally_length"), errors="coerce")
            masks["medium"] = ((rl >= 5) & (rl <= 8)).fillna(False) if rl is not None else pd.Series([False]*len(use), index=use.index)
        if "rally_9" in use.columns:
            masks["long"] = _to_bool_series(use["rally_9"])
        else:
            rl = pd.to_numeric(use.get("rally_length"), errors="coerce")
            masks["long"] = (rl >= 9).fillna(False) if rl is not None else pd.Series([False]*len(use), index=use.index)
        return masks

    rally_masks = _rally_bucket_masks()

    def _win_pct(mask: pd.Series, pid: str) -> Optional[float]:
        pts = use[mask]
        if len(pts) == 0:
            return None
        return _pct(float((pts["pt_won_by"] == pid).mean()))

    def _count_end(col: str, pid: str, mask: Optional[pd.Series] = None) -> int:
        if col not in use.columns:
            return 0
        s = use[col]
        if mask is not None:
            s = s[mask]
        return int((s == pid).sum())

    def _fault_profile(pid: str) -> Dict[str, Any]:
        pts = use[use["server"] == pid]
        if len(pts) == 0 or "1st_serve" not in pts.columns:
            return {}
        s1 = pts["1st_serve"]
        s2 = pts["2nd_serve"] if "2nd_serve" in pts.columns else pd.Series([pd.NA]*len(pts), index=pts.index)

        s1_fault = s1.apply(_is_fault)
        s2_fault = s2.apply(_is_fault)

        def ft_counts(series: pd.Series, fault_mask: pd.Series) -> Dict[str, int]:
            vals = series[fault_mask].apply(_fault_type).dropna()
            return {k: int(v) for k, v in vals.value_counts().to_dict().items()}

        return {
            "first_serve_fault_rate_pct": _pct(float(s1_fault.mean())),
            "first_serve_fault_type_counts": ft_counts(s1, s1_fault),
            "second_serve_fault_rate_pct": _pct(float(s2_fault.mean())) if s2.notna().any() else None,
            "second_serve_fault_type_counts": ft_counts(s2, s2_fault) if s2.notna().any() else {},
        }

    def _serve_dir_entropy(pid: str, mask: Optional[pd.Series] = None) -> Optional[float]:
        pts = use[use["server"] == pid]
        if mask is not None:
            pts = pts[mask.loc[pts.index]]
        if len(pts) == 0:
            return None
        counts = pts["final_serve_dir"].fillna("UNKNOWN").value_counts()
        total = float(counts.sum())
        probs = [float(c/total) for c in counts.values] if total > 0 else []
        return _entropy_norm(probs)

    def _serve_dir_roi(pid: str, mask: Optional[pd.Series] = None) -> Dict[str, Any]:
        pts = use[use["server"] == pid]
        if mask is not None:
            pts = pts[mask.loc[pts.index]]
        if len(pts) == 0:
            return {}
        base = float((pts["pt_won_by"] == pid).mean())
        out = {}
        for d, g in pts.groupby(pts["final_serve_dir"].fillna("UNKNOWN")):
            win = float((g["pt_won_by"] == pid).mean()) if len(g) else np.nan
            out[str(d)] = {
                "points": int(len(g)),
                "win_pct": _pct(win) if not np.isnan(win) else None,
                "roi_pp": float(np.round((win - base) * 100.0, 2)) if not np.isnan(win) else None,
            }
        return out

    def _side_split(pid: str) -> Dict[str, Any]:
        if "sides" not in use.columns:
            return {}
        out = {}
        for side in ["deuce", "adv"]:
            m = (use["sides"] == side)
            out[side] = {
                "overall_win_pct": _win_pct(m, pid),
                "serve_win_pct": _win_pct(m & (use["server"] == pid), pid),
                "return_win_pct": _win_pct(m & (use["server"] != pid), pid),
                "points": int(m.sum()),
            }
        return out

    def _clutch_stats(pid: str) -> Dict[str, Any]:
        out = {}
        # Overall clutch union
        m_all = use["is_clutch_union"]
        out["clutch_union"] = {
            "points": int(m_all.sum()),
            "win_pct": _win_pct(m_all, pid),
            "errors": _count_end("error", pid, m_all),
            "double_faults": int(((use.get("double_fault") == pid) & m_all).sum()) if "double_fault" in use.columns else 0,
        }
        # Specific score states
        for key, col in [("30_30", "is_30_30"), ("deuce", "is_deuce"), ("ad_point", "is_ad_point")]:
            m = use[col]
            out[key] = {"points": int(m.sum()), "win_pct": _win_pct(m, pid)}
        # Pressure union split
        m_p = use["is_pressure_union"]
        out["pressure_union"] = {"points": int(m_p.sum()), "win_pct": _win_pct(m_p, pid)}
        return out

    def _first_strike(pid: str) -> Dict[str, Any]:
        short = rally_masks["short"]
        return {
            "serve_short_rally_win_pct": _win_pct(short & (use["server"] == pid), pid),
            "return_short_rally_win_pct": _win_pct(short & (use["server"] != pid), pid),
        }

    def _return_vs_serve_dir(pid: str) -> Dict[str, Any]:
        pts = use[use["server"] != pid]
        if len(pts) == 0:
            return {}
        out = {}
        for d, g in pts.groupby(pts["final_serve_dir"].fillna("UNKNOWN")):
            out[str(d)] = {"points": int(len(g)), "win_pct": _pct(float((g["pt_won_by"] == pid).mean()))}
        return out

    def _winners_errors_by_bucket(pid: str) -> Dict[str, Any]:
        out = {}
        for b, m in rally_masks.items():
            pts = use[m]
            out[b] = {
                "points": int(len(pts)),
                "win_pct": _win_pct(m, pid),
                "winners": _count_end("winner", pid, m),
                "errors": _count_end("error", pid, m),
                "double_faults": int(((use.get("double_fault") == pid) & m).sum()) if "double_fault" in use.columns else 0,
                "winner_minus_error": _count_end("winner", pid, m) - _count_end("error", pid, m),
            }
        return out

    def _error_context(pid: str) -> Dict[str, Any]:
        """
        Error/winner rates by:
          role (serve/return) x rally_bucket (short/medium/long) x pressure_union (yes/no)
        """
        out: Dict[str, Any] = {}
        for role in ["serve", "return"]:
            role_m = _context_mask(role, pid)
            out[role] = {}
            for bucket, b_m in rally_masks.items():
                out[role][bucket] = {}
                for pressure in ["pressure", "non_pressure"]:
                    p_m = use["is_pressure_union"] if pressure == "pressure" else ~use["is_pressure_union"]
                    m = role_m & b_m & p_m
                    pts = use[m]
                    n = int(len(pts))
                    winners = _count_end("winner", pid, m)
                    errors = _count_end("error", pid, m)
                    dfs = int(((use.get("double_fault") == pid) & m).sum()) if "double_fault" in use.columns else 0
                    out[role][bucket][pressure] = {
                        "points": n,
                        "win_pct": None if n == 0 else _pct(float((pts["pt_won_by"] == pid).mean())),
                        "winners": winners,
                        "errors": errors,
                        "double_faults": dfs,
                        "winner_minus_error": winners - errors,
                        "error_rate_pct": None if n == 0 else _pct((errors + dfs) / n),
                    }
        return out

    def _double_fault_severity(pid: str) -> Dict[str, Any]:
        if "double_fault" not in use.columns:
            return {"count": 0, "severity_sum": 0, "severity_avg": None}

        df_mask = (use["double_fault"] == pid)
        if int(df_mask.sum()) == 0:
            return {"count": 0, "severity_sum": 0, "severity_avg": None}

        # Severity weights derived from existing flags (no invented signals).
        w = pd.Series(1.0, index=use.index)
        w = w + use["is_30_30"].astype(float) + use["is_deuce"].astype(float) + use["is_ad_point"].astype(float)
        w = w + (use["is_pressure_union"].astype(float) * 2.0)
        severity = float((w[df_mask]).sum())
        return {
            "count": int(df_mask.sum()),
            "severity_sum": float(np.round(severity, 2)),
            "severity_avg": float(np.round(severity / int(df_mask.sum()), 2)) if int(df_mask.sum()) else None,
        }

    def _error_speed(pid: str) -> Dict[str, Any]:
        if "error" not in use.columns:
            return {}
        e = use[use["error"] == pid]
        total = int(len(e))
        if total == 0:
            return {"errors": 0}
        # Share of errors in short rallies
        short_e = int((rally_masks["short"] & (use["error"] == pid)).sum())
        # Duration distribution (if available)
        if "point_duration" in use.columns:
            dur = pd.to_numeric(e["point_duration"], errors="coerce").dropna().astype(float).values
            if len(dur):
                p25, p50, p75 = np.percentile(dur, [25, 50, 75]).tolist()
                dur_stats = {
                    "point_duration_avg_s": float(np.round(dur.mean(), 2)),
                    "point_duration_p25_s": float(np.round(p25, 2)),
                    "point_duration_p50_s": float(np.round(p50, 2)),
                    "point_duration_p75_s": float(np.round(p75, 2)),
                }
            else:
                dur_stats = {}
        else:
            dur_stats = {}

        return {
            "errors": total,
            "errors_in_short_rallies_pct": _pct(short_e / total) if total else None,
            **dur_stats,
        }

    def _rally_style(pid: str) -> Dict[str, Any]:
        short = rally_masks["short"]
        long = rally_masks["long"]
        short_win = _win_pct(short, pid)
        long_win = _win_pct(long, pid)
        # tolerance score = long win% - short win% (pp)
        tol = None
        if short_win is not None and long_win is not None:
            tol = float(np.round(long_win - short_win, 2))
        # patience index = errors per long rally
        long_pts = int(long.sum())
        long_errors = _count_end("error", pid, long)
        patience = None if long_pts == 0 else float(np.round(long_errors / long_pts, 4))
        return {"rally_tolerance_pp": tol, "patience_index_errors_per_long_point": patience}

    def _rally_length_stats(pid: str) -> Dict[str, Any]:
        if "rally_length" not in use.columns:
            return {}
        rl = pd.to_numeric(use["rally_length"], errors="coerce").dropna().astype(float)
        if len(rl) == 0:
            return {}
        # conditional when player wins/loses
        win_rl = pd.to_numeric(use.loc[use["pt_won_by"] == pid, "rally_length"], errors="coerce").dropna().astype(float)
        lose_rl = pd.to_numeric(use.loc[use["pt_won_by"] != pid, "rally_length"], errors="coerce").dropna().astype(float)
        def _summ(x: pd.Series) -> Dict[str, Any]:
            if len(x) == 0:
                return {}
            return {
                "mean": float(np.round(x.mean(), 2)),
                "median": float(np.round(x.median(), 2)),
                "std": float(np.round(x.std(ddof=0), 2)),
            }
        return {
            "overall": _summ(rl),
            "when_winning": _summ(win_rl),
            "when_losing": _summ(lose_rl),
        }

    def _net_roi(pid: str) -> Dict[str, Any]:
        base_win = float((use["pt_won_by"] == pid).mean())
        net = _platform_net_points(pid)
        net_win = None
        if net["played"] > 0:
            net_win = net["won"] / net["played"]
        roi_pp = None if net_win is None else float(np.round((net_win - base_win) * 100.0, 2))

        # net under pressure union
        col = "p1_net_appearance" if pid == p1_id else "p2_net_appearance"
        if col in use.columns:
            net_mask = _to_bool_series(use[col])
            m = net_mask & use["is_pressure_union"]
            pts = use[m]
            net_pressure_win = None if len(pts) == 0 else float((pts["pt_won_by"] == pid).mean())
            net_pressure_win_pct = None if net_pressure_win is None else _pct(net_pressure_win)
        else:
            net_pressure_win_pct = None

        return {
            "net_roi_pp": roi_pp,
            "net_pressure_win_pct": net_pressure_win_pct,
        }

    def _flow(pid: str) -> Dict[str, Any]:
        """
        Point streaks + game-level hold/break & break-back/consolidation where possible.
        """
        out: Dict[str, Any] = {}

        # Point streaks
        winners_seq = use["pt_won_by"].astype(str).tolist()
        max_win, max_lose = 0, 0
        cur = 0
        for w in winners_seq:
            if w == pid:
                cur += 1
                max_win = max(max_win, cur)
            else:
                cur = 0
        cur = 0
        for w in winners_seq:
            if w != pid:
                cur += 1
                max_lose = max(max_lose, cur)
            else:
                cur = 0
        out["point_streaks"] = {"max_points_won_in_row": int(max_win), "max_points_lost_in_row": int(max_lose)}

        # Game-level metrics
        games = _extract_games(use, p1_id, p2_id)
        if not games:
            return out

        # Identify service games played and won
        service_games = [g for g in games if g["server_id"] == pid]
        return_games = [g for g in games if g["server_id"] != pid]
        holds = [g for g in service_games if g["winner_id"] == pid]
        breaks = [g for g in return_games if g["winner_id"] == pid]

        out["games"] = {
            "service_games_played": int(len(service_games)),
            "service_games_won": int(len(holds)),
            "hold_rate_pct": None if len(service_games) == 0 else _pct(len(holds) / len(service_games)),
            "return_games_played": int(len(return_games)),
            "return_games_won": int(len(breaks)),
            "break_rate_pct": None if len(return_games) == 0 else _pct(len(breaks) / len(return_games)),
        }

        # Game streaks
        winners = [g["winner_id"] for g in games if g["winner_id"] is not None]
        max_gw = 0
        cur = 0
        for w in winners:
            if w == pid:
                cur += 1
                max_gw = max(max_gw, cur)
            else:
                cur = 0
        out["game_streaks"] = {"max_games_won_in_row": int(max_gw)}

        # Break-back & consolidation
        break_back_opps = 0
        break_backs = 0
        consolidation_opps = 0
        consolidations = 0

        for i, g in enumerate(games[:-1]):
            # If player got broken (lost service game), next return game is an opportunity to break back
            if g["server_id"] == pid and g["winner_id"] != pid:
                # find next game where pid is returner (likely next game, but handle if not)
                for j in range(i + 1, len(games)):
                    if games[j]["server_id"] != pid:
                        break_back_opps += 1
                        if games[j]["winner_id"] == pid:
                            break_backs += 1
                        break
            # If player broke (won return game), next service game is an opportunity to consolidate
            if g["server_id"] != pid and g["winner_id"] == pid:
                for j in range(i + 1, len(games)):
                    if games[j]["server_id"] == pid:
                        consolidation_opps += 1
                        if games[j]["winner_id"] == pid:
                            consolidations += 1
                        break

        out["break_back"] = {
            "opportunities": int(break_back_opps),
            "converted": int(break_backs),
            "rate_pct": None if break_back_opps == 0 else _pct(break_backs / break_back_opps),
        }
        out["consolidation"] = {
            "opportunities": int(consolidation_opps),
            "converted": int(consolidations),
            "rate_pct": None if consolidation_opps == 0 else _pct(consolidations / consolidation_opps),
        }

        return out

    def _turning_points() -> Dict[str, Any]:
        """
        Rank candidate turning points for narrative insights.
        Purely derived from existing flags.
        """
        points = []
        # Long rallies
        if "rally_length" in use.columns:
            long = use.sort_values("rally_length", ascending=False).head(10)
        else:
            long = use.head(0)

        long_ids = set(long["point_idx"].astype(int).tolist())

        for _, r in use.iterrows():
            pid = int(r["point_idx"])
            tags = []
            for c in ["matchpoint", "setpoint", "breakpoint", "gamepoint", "pressure_point"]:
                if c in use.columns and pd.notna(r.get(c)):
                    tags.append(c)
            if bool(r.get("highlights", False)) and "highlights" in use.columns:
                tags.append("highlights")
            if pid in long_ids:
                tags.append("long_rally_top10")
            if not tags:
                continue
            points.append({
                "point_idx": pid,
                "start_s": float(r.get("start_s", np.nan)),
                "end_s": float(r.get("end_s", np.nan)),
                "server": str(r.get("server")),
                "pt_won_by": str(r.get("pt_won_by")),
                "tags": tags,
                "rally_length": None if pd.isna(r.get("rally_length")) else int(r.get("rally_length")),
                "rally_summary": "" if pd.isna(r.get("rally_summary")) else str(r.get("rally_summary")),
            })

        # Sort: matchpoint > setpoint > breakpoint > others, then long rallies
        priority = {"matchpoint": 5, "setpoint": 4, "breakpoint": 3, "gamepoint": 2, "pressure_point": 1, "highlights": 1, "long_rally_top10": 1}
        def score(p):
            return sum(priority.get(t, 0) for t in p["tags"]), (p["rally_length"] or 0)
        points = sorted(points, key=score, reverse=True)[:30]
        return {"points": points}

    def _tempo() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if "point_duration" in use.columns:
            dur = pd.to_numeric(use["point_duration"], errors="coerce").dropna().astype(float)
            if len(dur):
                out["point_duration"] = {
                    "avg_s": float(np.round(dur.mean(), 2)),
                    "median_s": float(np.round(dur.median(), 2)),
                    "p75_s": float(np.round(np.percentile(dur, 75), 2)),
                }
                # duration by end_type
                if "end_type" in use.columns:
                    out["point_duration_by_end_type"] = {}
                    for et, g in use.groupby(use["end_type"].fillna("OTHER")):
                        gd = pd.to_numeric(g["point_duration"], errors="coerce").dropna().astype(float)
                        if len(gd):
                            out["point_duration_by_end_type"][str(et)] = {"points": int(len(g)), "avg_s": float(np.round(gd.mean(), 2))}
        # Rest gap between points (requires start/end)
        if "start_s" in use.columns and "end_s" in use.columns:
            starts = pd.to_numeric(use["start_s"], errors="coerce").values
            ends = pd.to_numeric(use["end_s"], errors="coerce").values
            gaps = []
            for i in range(len(use) - 1):
                if not (math.isnan(starts[i+1]) or math.isnan(ends[i])):
                    gap = starts[i+1] - ends[i]
                    if gap >= 0:
                        gaps.append(gap)
            if gaps:
                gaps = np.array(gaps, dtype=float)
                out["between_points_gap_s"] = {
                    "avg_s": float(np.round(gaps.mean(), 2)),
                    "median_s": float(np.round(np.median(gaps), 2)),
                    "p75_s": float(np.round(np.percentile(gaps, 75), 2)),
                }
        return out

    def _momentum() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for col, key in [("p1_momentum", "p1"), ("p2_momentum", "p2")]:
            if col not in use.columns:
                continue
            vals = pd.to_numeric(use[col], errors="coerce").dropna().astype(float).values
            if len(vals) < 2:
                continue
            diffs = np.diff(vals)
            out[key] = {
                "std": float(np.round(vals.std(ddof=0), 4)),
                "avg_abs_delta": float(np.round(np.mean(np.abs(diffs)), 4)),
                "max_abs_delta": float(np.round(np.max(np.abs(diffs)), 4)),
            }
        return out

    def _decision_drivers() -> Dict[str, Any]:
        """
        Produce a ranked list of the biggest stat gaps between players.
        This is helpful for LLM: "Top 3 reasons you won/lost".
        """
        drivers = []

        # Helper to add driver with magnitude and context
        def add(name: str, p1_val: Optional[float], p2_val: Optional[float], unit: str):
            if p1_val is None or p2_val is None:
                return
            gap = float(np.round((p2_val - p1_val), 4))
            drivers.append({"name": name, "p1": p1_val, "p2": p2_val, "gap_p2_minus_p1": gap, "unit": unit})

        # Use already computed metrics where possible
        p1_pw = p1_points_won / total_points if total_points else None
        p2_pw = p2_points_won / total_points if total_points else None
        add("points_won_pct", p1_pw, p2_pw, "ratio")

        sb1 = serve_block(p1_id)
        sb2 = serve_block(p2_id)
        add("service_points_won_pct", sb1.get("service_points_won_pct", None)/100 if sb1.get("service_points_won_pct") is not None else None,
            sb2.get("service_points_won_pct", None)/100 if sb2.get("service_points_won_pct") is not None else None, "ratio")
        add("second_serve_points_won_pct", sb1.get("second_serve_points_won_pct", None)/100 if sb1.get("second_serve_points_won_pct") is not None else None,
            sb2.get("second_serve_points_won_pct", None)/100 if sb2.get("second_serve_points_won_pct") is not None else None, "ratio")

        rb1 = return_block(p1_id)
        rb2 = return_block(p2_id)
        add("return_points_won_pct", rb1.get("return_points_won_pct", None)/100 if rb1.get("return_points_won_pct") is not None else None,
            rb2.get("return_points_won_pct", None)/100 if rb2.get("return_points_won_pct") is not None else None, "ratio")

        # Errors and winners gaps (counts)
        add("errors_count", float(_platform_unforced_errors(p1_id)), float(_platform_unforced_errors(p2_id)), "count")
        add("winners_count", float(_platform_winners(p1_id)), float(_platform_winners(p2_id)), "count")

        # Breakpoint conversion gaps (ratio)
        bp1 = platform_visible["break_points"]["p1"]["conversion_pct"]
        bp2 = platform_visible["break_points"]["p2"]["conversion_pct"]
        add("breakpoint_conversion_pct", None if bp1 is None else bp1/100.0, None if bp2 is None else bp2/100.0, "ratio")

        # Rally bucket win gaps
        for b in ["short", "long"]:
            w1 = _win_pct(rally_masks[b], p1_id)
            w2 = _win_pct(rally_masks[b], p2_id)
            add(f"{b}_rally_win_pct", None if w1 is None else w1/100.0, None if w2 is None else w2/100.0, "ratio")

        # Rank by absolute gap (counts and ratios are mixed; still useful for LLM ordering)
        drivers_sorted = sorted(drivers, key=lambda d: abs(d["gap_p2_minus_p1"]), reverse=True)
        return {"drivers_ranked": drivers_sorted[:10]}

    # Build extended stats per player
    pressure_mask = use["is_pressure_union"]

    llm_extended = {
        "clutch": {"p1": _clutch_stats(p1_id), "p2": _clutch_stats(p2_id)},
        "side_splits": {"p1": _side_split(p1_id), "p2": _side_split(p2_id)},
        "serve_fault_profile": {"p1": _fault_profile(p1_id), "p2": _fault_profile(p2_id)},
        "serve_direction_entropy": {
            "overall": {"p1": _serve_dir_entropy(p1_id), "p2": _serve_dir_entropy(p2_id)},
            "pressure_union": {"p1": _serve_dir_entropy(p1_id, pressure_mask), "p2": _serve_dir_entropy(p2_id, pressure_mask)},
        },
        "serve_direction_roi": {
            "overall": {"p1": _serve_dir_roi(p1_id), "p2": _serve_dir_roi(p2_id)},
            "pressure_union": {"p1": _serve_dir_roi(p1_id, pressure_mask), "p2": _serve_dir_roi(p2_id, pressure_mask)},
        },
        "first_strike": {"p1": _first_strike(p1_id), "p2": _first_strike(p2_id)},
        "return_vs_serve_direction": {"p1": _return_vs_serve_dir(p1_id), "p2": _return_vs_serve_dir(p2_id)},
        "winners_errors_by_rally_bucket": {"p1": _winners_errors_by_bucket(p1_id), "p2": _winners_errors_by_bucket(p2_id)},
        "error_context": {"p1": _error_context(p1_id), "p2": _error_context(p2_id)},
        "double_fault_severity": {"p1": _double_fault_severity(p1_id), "p2": _double_fault_severity(p2_id)},
        "error_speed": {"p1": _error_speed(p1_id), "p2": _error_speed(p2_id)},
        "rally_style": {"p1": _rally_style(p1_id), "p2": _rally_style(p2_id)},
        "rally_length_stats": {"p1": _rally_length_stats(p1_id), "p2": _rally_length_stats(p2_id)},
        "net_roi": {"p1": _net_roi(p1_id), "p2": _net_roi(p2_id)},
        "flow": {"p1": _flow(p1_id), "p2": _flow(p2_id)},
        "turning_points": _turning_points(),
        "tempo": _tempo(),
        "momentum_volatility": _momentum(),
        "decision_drivers": _decision_drivers(),
    }

    # ---------------------------
    # Final return object (compatibility + new blocks)
    # ---------------------------

    return {
        "players": {
            "p1": {"id": p1_id, "name": p1_name},
            "p2": {"id": p2_id, "name": p2_name},
        },
        "totals": {
            "total_points": total_points,
            "points_won": {"p1": p1_points_won, "p2": p2_points_won},
            "points_won_pct": {
                "p1": _pct(p1_points_won / total_points) if total_points else 0.0,
                "p2": _pct(p2_points_won / total_points) if total_points else 0.0,
            },
            # Timing is useful even if not shown in UI
            "active_time_s": active_time_s,
            "removed_time_s": removed_time_s,
            "span_s": span_s,
        },
        "kpis": {
            # Compatibility (existing app references these)
            "serve": {"p1": serve_block(p1_id), "p2": serve_block(p2_id)},
            "return": {"p1": return_block(p1_id), "p2": return_block(p2_id)},

            # Demarcation: current platform match page stats
            "platform_visible": platform_visible,

            # Demarcation: extra LLM-only stats (not shown on platform for now)
            "llm_extended": llm_extended,
        },
    }
