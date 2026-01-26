from __future__ import annotations
from typing import Any, Dict
import numpy as np
import pandas as pd

def _pct(x: float) -> float:
    return float(np.round(x * 100.0, 2))

def compute_match_stats(df: pd.DataFrame) -> Dict[str, Any]:
    use = df[df["discard_point"] != True].copy()  # noqa: E712

    p1_id = str(use.iloc[0]["p1_id"])
    p2_id = str(use.iloc[0]["p2_id"])
    p1_name = str(use.iloc[0]["p1_fullname"]).strip()
    p2_name = str(use.iloc[0]["p2_fullname"]).strip()

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

        df_count = int((pts.get("double_fault") == pid).sum()) if "double_fault" in pts.columns else 0

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

    total_points = int(len(use))
    p1_points_won = int((use["pt_won_by"] == p1_id).sum())
    p2_points_won = int((use["pt_won_by"] == p2_id).sum())

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
        },
        "kpis": {
            "serve": {"p1": serve_block(p1_id), "p2": serve_block(p2_id)},
            "return": {"p1": return_block(p1_id), "p2": return_block(p2_id)},
        },
    }
