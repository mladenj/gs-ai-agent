from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict
import pandas as pd

REQUIRED_COLS = [
    "p1_id", "p2_id", "p1_fullname", "p2_fullname",
    "server", "pt_won_by",
    "vid_second", "end",
    "rally_length",
    "rally_summary", "rally_desc",
    "discard_point",
]

def _parse_serve(text: Any) -> Dict[str, Any]:
    if pd.isna(text):
        return {"dir": None, "fault": False, "fault_type": None}
    raw = str(text).lower()
    dir_ = None
    if "down the t" in raw:
        dir_ = "T"
    elif "to body" in raw:
        dir_ = "BODY"
    elif "wide" in raw:
        dir_ = "WIDE"
    fault = "fault" in raw
    ft = None
    if fault:
        m = re.search(r"fault\s*\(([^)]+)\)", raw)
        if m:
            ft = m.group(1).strip().upper()
    return {"dir": dir_, "fault": fault, "fault_type": ft}

def normalize_points(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Expected columns: {REQUIRED_COLS}. "
            f"Check if your CSV export format has changed or if column names differ."
        )

    out = df.copy()

    out["start_s"] = pd.to_numeric(out["vid_second"], errors="coerce")
    out["end_s"] = pd.to_numeric(out["end"], errors="coerce")

    s1 = out["1st_serve"] if "1st_serve" in out.columns else pd.Series([pd.NA] * len(out))
    s2 = out["2nd_serve"] if "2nd_serve" in out.columns else pd.Series([pd.NA] * len(out))

    out["serve_num_used"] = (~s2.isna()).astype(int).replace({0: 1, 1: 2})

    s1p = s1.apply(_parse_serve)
    s2p = s2.apply(_parse_serve)

    out["s1_fault"] = s1p.apply(lambda d: bool(d["fault"]))
    out["s2_fault"] = s2p.apply(lambda d: bool(d["fault"]))

    out["final_serve_dir"] = pd.Series(
        [
            (d2["dir"] if not pd.isna(s2.iloc[i]) else d1["dir"])
            for i, (d1, d2) in enumerate(zip(s1p, s2p))
        ]
    ).fillna("UNKNOWN")

    out["end_type"] = "OTHER"
    if "double_fault" in out.columns:
        out.loc[out["double_fault"].notna(), "end_type"] = "DOUBLE_FAULT"
    if "winner" in out.columns:
        out.loc[out["winner"].notna(), "end_type"] = "WINNER"
    if "error" in out.columns:
        out.loc[out["error"].notna(), "end_type"] = "ERROR"

    out = out.reset_index(drop=True)
    out["point_idx"] = out.index + 1
    return out

def load_l2_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_points(df)
