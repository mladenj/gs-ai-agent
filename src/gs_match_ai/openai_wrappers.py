from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI

from .schemas import INSIGHTS_JSON_SCHEMA, MULTI_INSIGHTS_JSON_SCHEMA


# ---------------------------------------------------------------------------
# Model defaults (no UI selection; overridable via env/secrets)
# ---------------------------------------------------------------------------

def _default_insight_model() -> str:
    return (
        os.getenv("OPENAI_MODEL_INSIGHTS")
        or os.getenv("OPENAI_MODEL")
        or "gpt-5-mini"
    )


def _default_chat_model() -> str:
    return (
        os.getenv("OPENAI_MODEL_CHAT")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Evidence shaping helpers (quality-first, minimal trimming)
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types so json.dumps doesn't fail."""
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore
        if obj is pd.NA:
            return None
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


def make_llm_evidence_view(
    evidence_packet: Dict[str, Any],
    *,
    max_list_items: int = 200,
    max_str_len: int = 8000,
) -> Dict[str, Any]:
    """Return an LLM-friendly view of the evidence packet.

    We keep all computed stats, but cap very long lists/strings so prompts remain stable.
    This is intentionally conservative (quality-first, not aggressive pre-optimization).
    """
    ep = _to_json_safe(evidence_packet)

    def _trim(x: Any) -> Any:
        if isinstance(x, str) and len(x) > max_str_len:
            return x[:max_str_len] + "…"
        if isinstance(x, list) and len(x) > max_list_items:
            return x[:max_list_items]
        if isinstance(x, dict):
            return {k: _trim(v) for k, v in x.items()}
        return x

    return _trim(ep)


# ---------------------------------------------------------------------------
# Responses API extraction + repair helpers
# ---------------------------------------------------------------------------

def _extract_response_text(resp: Any) -> str:
    """Extract text from an OpenAI Responses API response.

    Some models may return empty `output_text` even when content exists in
    `resp.output[*].content[*].text`. This helper makes extraction robust.
    """
    raw = (getattr(resp, "output_text", None) or "").strip()
    if raw:
        return raw

    parts: list[str] = []
    for item in (getattr(resp, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            t = getattr(c, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t)
    return "\n".join(parts).strip()


def _parse_json_or_raise(raw: str, *, model: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError(
            f"Empty response text from model={model}. "
            "This usually means the model returned no textual content."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        preview = raw[:500].replace("\n", " ")
        raise ValueError(
            f"Model={model} returned non-JSON or truncated JSON. "
            f"JSONDecodeError={e}. Preview='{preview}...'"
        ) from e


def _repair_instruction() -> str:
    return (
        "REPAIR MODE: Your previous output was invalid. "
        "Return ONLY valid JSON that matches the provided JSON schema strictly. "
        "Do not include any extra text, markdown, code fences, or commentary."
    )




# ---------------------------------------------------------------------------
# Trend block computation (precomputed, authoritative, to prevent LLM math errors)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Better-when semantics for each metric
# ---------------------------------------------------------------------------
_BETTER_WHEN: dict[str, str] = {
    # --- higher is better ---
    "points_won_pct":                    "higher",
    "first_serve_pct":                   "higher",
    "first_serve_points_won_pct":        "higher",
    "second_serve_points_won_pct":       "higher",
    "return_points_won_pct":             "higher",
    "return_vs_second_serve_win_pct":    "higher",
    "break_points_conversion_pct":       "higher",
    "pressure_points_win_pct":           "higher",
    "net_points_win_pct":                "higher",
    "short_rally_win_pct":               "higher",
    "medium_rally_win_pct":              "higher",
    "long_rally_win_pct":                "higher",
    "winners":                           "higher",
    "clutch_deuce_win_pct":              "higher",
    "clutch_ad_win_pct":                 "higher",
    "clutch_pressure_union_win_pct":     "higher",
    "side_deuce_win_pct":                "higher",
    "side_ad_win_pct":                   "higher",
    "flow_break_back_rate":              "higher",
    "flow_consolidation_rate":           "higher",
    "rally_tolerance_score":             "higher",
    # --- lower is better ---
    "double_faults":                     "lower",
    "unforced_errors":                   "lower",
    "pressure_error_share":              "lower",
    # --- contextual ---
    "serve_predictability_entropy":      "contextual",
}

# ---------------------------------------------------------------------------
# Metric extraction spec
# Each entry: (label, unit, value_extractor, sample_extractor | None)
# value_extractor: fn(ep, pk) -> float | None   (ep = evidence_packet, pk = "p1"|"p2")
# sample_extractor: fn(ep, pk) -> int | None
# ---------------------------------------------------------------------------

def _ep_get(ep: dict, *keys: str) -> Any:
    """Safely navigate nested dict; returns None if any key is missing."""
    cur: Any = ep
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _build_metric_specs() -> list[dict]:
    """
    Returns a list of metric spec dicts:
      metric_id, label, unit, better_when, value_fn, sample_fn
    """

    def _pv(ep: dict, pk: str, *path: str) -> Any:
        """Shortcut: ep -> kpis.platform_visible -> path"""
        return _ep_get(ep, "kpis", "platform_visible", *path)

    def _lx(ep: dict, pk: str, *path: str) -> Any:
        """Shortcut: ep -> kpis.llm_extended -> path"""
        return _ep_get(ep, "kpis", "llm_extended", *path)

    specs = [
        # ---- Platform-visible ----
        {
            "metric_id": "points_won_pct",
            "label":     "Points Won %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "points_won", "points", "won_pct", pk),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "points_won", "points", "total"),
        },
        {
            "metric_id": "first_serve_pct",
            "label":     "First Serve In %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "first_serve_in_pct", pk, "in_pct"),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "first_serve_in_pct", pk, "attempts"),
        },
        {
            "metric_id": "winners",
            "label":     "Winners",
            "unit":      "count",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "winners", pk),
            "sample_fn": None,
        },
        {
            "metric_id": "unforced_errors",
            "label":     "Unforced Errors",
            "unit":      "count",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "unforced_errors", pk),
            "sample_fn": None,
        },
        {
            "metric_id": "double_faults",
            "label":     "Double Faults",
            "unit":      "count",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "double_faults", pk),
            "sample_fn": None,
        },
        {
            "metric_id": "break_points_conversion_pct",
            "label":     "Breakpoint Conversion %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "break_points", pk, "conversion_pct"),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "break_points", pk, "earned"),
        },
        {
            "metric_id": "pressure_points_win_pct",
            "label":     "Pressure Points Won %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "pressure_points", pk, "won_pct"),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "pressure_points", pk, "played"),
        },
        {
            "metric_id": "net_points_win_pct",
            "label":     "Net Points Won %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _pv(ep, pk, "net_points", pk, "won_pct"),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "net_points", pk, "played"),
        },
        {
            "metric_id": "short_rally_win_pct",
            "label":     "Short Rally Win % (1-4 shots)",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: (
                # Compute from wins/points in the bucket for the focus player
                _safe_pct(
                    _pv(ep, pk, "rally_analysis", "short", "wins", pk),
                    _pv(ep, pk, "rally_analysis", "short", "points"),
                )
            ),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "rally_analysis", "short", "points"),
        },
        {
            "metric_id": "medium_rally_win_pct",
            "label":     "Medium Rally Win % (5-8 shots)",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: (
                _safe_pct(
                    _pv(ep, pk, "rally_analysis", "medium", "wins", pk),
                    _pv(ep, pk, "rally_analysis", "medium", "points"),
                )
            ),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "rally_analysis", "medium", "points"),
        },
        {
            "metric_id": "long_rally_win_pct",
            "label":     "Long Rally Win % (9+ shots)",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: (
                _safe_pct(
                    _pv(ep, pk, "rally_analysis", "long", "wins", pk),
                    _pv(ep, pk, "rally_analysis", "long", "points"),
                )
            ),
            "sample_fn": lambda ep, pk: _pv(ep, pk, "rally_analysis", "long", "points"),
        },
        # ---- LLM-extended ----
        {
            "metric_id": "clutch_deuce_win_pct",
            "label":     "Deuce Point Win %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "clutch", pk, "deuce", "win_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "clutch", pk, "deuce", "points"),
        },
        {
            "metric_id": "clutch_ad_win_pct",
            "label":     "Ad Point Win %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "clutch", pk, "ad_point", "win_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "clutch", pk, "ad_point", "points"),
        },
        {
            "metric_id": "clutch_pressure_union_win_pct",
            "label":     "Pressure Union Win %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "clutch", pk, "pressure_union", "win_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "clutch", pk, "pressure_union", "points"),
        },
        {
            "metric_id": "side_deuce_win_pct",
            "label":     "Deuce-Side Win %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "side_splits", pk, "deuce", "overall_win_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "side_splits", pk, "deuce", "points"),
        },
        {
            "metric_id": "side_ad_win_pct",
            "label":     "Ad-Side Win %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "side_splits", pk, "adv", "overall_win_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "side_splits", pk, "adv", "points"),
        },
        {
            "metric_id": "return_vs_second_serve_win_pct",
            "label":     "Return Win % vs Opp 2nd Serve",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _ep_get(ep, "kpis", "return", pk, "return_points_won_vs_opp_second_pct"),
            "sample_fn": None,
        },
        {
            "metric_id": "serve_predictability_entropy",
            "label":     "Serve Direction Entropy",
            "unit":      "ratio",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "serve_direction_entropy", "overall", pk),
            "sample_fn": None,
        },
        {
            "metric_id": "pressure_error_share",
            "label":     "Pressure Error Share",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _compute_pressure_error_share(ep, pk),
            "sample_fn": lambda ep, pk: _compute_pressure_error_share_sample(ep, pk),
        },
        {
            "metric_id": "rally_tolerance_score",
            "label":     "Rally Tolerance Score (long - short win pp)",
            "unit":      "ratio",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "rally_style", pk, "rally_tolerance_pp"),
            "sample_fn": None,
        },
        {
            "metric_id": "flow_break_back_rate",
            "label":     "Break-Back Rate %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "flow", pk, "break_back", "rate_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "flow", pk, "break_back", "opportunities"),
        },
        {
            "metric_id": "flow_consolidation_rate",
            "label":     "Consolidation Rate %",
            "unit":      "pct",
            "value_fn":  lambda ep, pk: _lx(ep, pk, "flow", pk, "consolidation", "rate_pct"),
            "sample_fn": lambda ep, pk: _lx(ep, pk, "flow", pk, "consolidation", "opportunities"),
        },
    ]
    return specs


def _safe_pct(wins: Any, total: Any) -> Optional[float]:
    """Compute wins/total * 100 safely; returns None if data not available."""
    try:
        w = float(wins)
        t = float(total)
        if t == 0:
            return None
        return round(w / t * 100.0, 2)
    except (TypeError, ValueError):
        return None


def _compute_pressure_error_share(ep: dict, pk: str) -> Optional[float]:
    """
    Pressure error share = errors under pressure / total errors.
    Derived from error_context: pressure errors across all buckets / all errors.
    """
    try:
        ec = _ep_get(ep, "kpis", "llm_extended", "error_context", pk)
        if not isinstance(ec, dict):
            return None
        pressure_errors = 0
        total_errors = 0
        for role_data in ec.values():
            if not isinstance(role_data, dict):
                continue
            for bucket_data in role_data.values():
                if not isinstance(bucket_data, dict):
                    continue
                p_data = bucket_data.get("pressure") or {}
                np_data = bucket_data.get("non_pressure") or {}
                pressure_errors += int(p_data.get("errors", 0) or 0)
                pressure_errors += int(p_data.get("double_faults", 0) or 0)
                total_errors += int(np_data.get("errors", 0) or 0)
                total_errors += int(np_data.get("double_faults", 0) or 0)
                total_errors += int(p_data.get("errors", 0) or 0)
                total_errors += int(p_data.get("double_faults", 0) or 0)
        if total_errors == 0:
            return None
        return round(pressure_errors / total_errors * 100.0, 2)
    except Exception:
        return None


def _compute_pressure_error_share_sample(ep: dict, pk: str) -> Optional[int]:
    try:
        ec = _ep_get(ep, "kpis", "llm_extended", "error_context", pk)
        if not isinstance(ec, dict):
            return None
        total_errors = 0
        for role_data in ec.values():
            if not isinstance(role_data, dict):
                continue
            for bucket_data in role_data.values():
                if not isinstance(bucket_data, dict):
                    continue
                for pres_key in ("pressure", "non_pressure"):
                    d = bucket_data.get(pres_key) or {}
                    total_errors += int(d.get("errors", 0) or 0)
                    total_errors += int(d.get("double_faults", 0) or 0)
        return total_errors if total_errors > 0 else None
    except Exception:
        return None


def _resolve_focus_player_key(ep: dict, focus_player: Dict[str, Any]) -> Optional[str]:
    """Return 'p1' or 'p2' based on focus_player id matching players block."""
    fp_id = str(focus_player.get("id", "")).strip()
    if not fp_id:
        return None
    p1_id = str(_ep_get(ep, "players", "p1", "id") or "").strip()
    p2_id = str(_ep_get(ep, "players", "p2", "id") or "").strip()
    if fp_id == p1_id:
        return "p1"
    if fp_id == p2_id:
        return "p2"
    # Try name fallback
    fp_name = str(focus_player.get("name", "")).strip().lower()
    p1_name = str(_ep_get(ep, "players", "p1", "name") or "").strip().lower()
    p2_name = str(_ep_get(ep, "players", "p2", "name") or "").strip().lower()
    if fp_name and fp_name == p1_name:
        return "p1"
    if fp_name and fp_name == p2_name:
        return "p2"
    return None


def _trend_direction_pct(delta_pp: float, flat_threshold: float = 1.0) -> str:
    if delta_pp > flat_threshold:
        return "increased"
    if delta_pp < -flat_threshold:
        return "decreased"
    return "flat"


def _trend_direction_count(delta: float) -> str:
    if delta > 1:
        return "increased"
    if delta < -1:
        return "decreased"
    return "flat"


def _materiality_pct(abs_delta_pp: float) -> tuple[str, str]:
    """Returns (materiality, interpretation_hint)."""
    if abs_delta_pp < 2.0:
        return "small", "stable"
    if abs_delta_pp <= 5.0:
        return "moderate", "possible shift"
    return "large", "clear change"


def _materiality_count(abs_delta: float) -> tuple[str, str]:
    if abs_delta <= 1:
        return "small", "stable"
    if abs_delta <= 3:
        return "moderate", "possible shift"
    return "large", "clear change"


def _performance_effect(direction: str, better_when: str) -> str:
    if direction == "flat":
        return "neutral"
    if better_when == "contextual":
        return "contextual"
    if better_when == "higher":
        return "improved" if direction == "increased" else "declined"
    if better_when == "lower":
        return "improved" if direction == "decreased" else "declined"
    return "contextual"


def _low_sample_threshold(metric_id: str, sample: Optional[int]) -> bool:
    if sample is None:
        return False
    # Break/pressure metrics: stricter threshold
    if any(k in metric_id for k in ("break_back", "consolidation", "pressure", "clutch", "bp")):
        return sample < 5
    return sample < 8


def _build_trend_blocks(
    matches_slim: List[Dict[str, Any]],
    focus_player: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build precomputed trend blocks comparing consecutive match pairs.

    Each block contains all fields needed for the LLM to describe trends without
    doing any arithmetic itself, preventing direction/wording mistakes.
    """
    metric_specs = _build_metric_specs()
    blocks: List[Dict[str, Any]] = []

    for i in range(len(matches_slim) - 1):
        m_a = matches_slim[i]
        m_b = matches_slim[i + 1]
        label_a = m_a.get("label", f"match_{i+1}")
        label_b = m_b.get("label", f"match_{i+2}")
        ep_a = m_a.get("evidence_packet", {})
        ep_b = m_b.get("evidence_packet", {})

        # Resolve focus player key (p1/p2) for each match independently
        pk_a = _resolve_focus_player_key(ep_a, focus_player)
        pk_b = _resolve_focus_player_key(ep_b, focus_player)
        # Fall back to p1 if we can't resolve (shouldn't happen, but be safe)
        if pk_a is None:
            pk_a = "p1"
        if pk_b is None:
            pk_b = "p1"

        for spec in metric_specs:
            metric_id: str = spec["metric_id"]
            label: str = spec["label"]
            unit: str = spec["unit"]
            better_when: str = _BETTER_WHEN.get(metric_id, "contextual")

            try:
                value_a = spec["value_fn"](ep_a, pk_a)
                value_b = spec["value_fn"](ep_b, pk_b)
            except Exception:
                continue

            if value_a is None or value_b is None:
                continue

            try:
                value_a = float(value_a)
                value_b = float(value_b)
            except (TypeError, ValueError):
                continue

            # Compute samples
            sample_a: Optional[int] = None
            sample_b: Optional[int] = None
            if spec.get("sample_fn"):
                try:
                    sa = spec["sample_fn"](ep_a, pk_a)
                    sb = spec["sample_fn"](ep_b, pk_b)
                    sample_a = int(sa) if sa is not None else None
                    sample_b = int(sb) if sb is not None else None
                except Exception:
                    pass

            low_sample = (
                _low_sample_threshold(metric_id, sample_a) or
                _low_sample_threshold(metric_id, sample_b)
            )

            # Compute delta, direction, materiality
            if unit in ("pct",):
                delta_pp = round(value_b - value_a, 2)
                delta = delta_pp
                direction = _trend_direction_pct(delta_pp)
                materiality, interpretation_hint = _materiality_pct(abs(delta_pp))
                delta_pp_out: Optional[float] = delta_pp
            elif unit == "count":
                delta = round(value_b - value_a, 2)
                delta_pp_out = None
                direction = _trend_direction_count(delta)
                materiality, interpretation_hint = _materiality_count(abs(delta))
            else:
                # ratio / sec: treat like pct for direction, materiality by abs delta
                delta_pp = round(value_b - value_a, 4)
                delta = delta_pp
                # For ratio, flat_threshold is smaller
                direction = _trend_direction_pct(delta_pp, flat_threshold=0.02)
                materiality, interpretation_hint = _materiality_pct(abs(delta_pp) * 100)
                delta_pp_out = delta_pp

            # Downgrade if low sample
            if low_sample:
                if interpretation_hint == "clear change":
                    interpretation_hint = "possible shift"
                if materiality == "large":
                    materiality = "moderate"

            performance_effect = _performance_effect(direction, better_when)

            blocks.append({
                "metric_id":          metric_id,
                "label":              label,
                "match_a_label":      label_a,
                "match_b_label":      label_b,
                "value_a":            round(value_a, 4),
                "value_b":            round(value_b, 4),
                "unit":               unit,
                "delta":              round(delta, 4),
                "delta_pp":           delta_pp_out,
                "direction":          direction,
                "better_when":        better_when,
                "performance_effect": performance_effect,
                "materiality":        materiality,
                "interpretation_hint": interpretation_hint,
                "sample_a":           sample_a,
                "sample_b":           sample_b,
                "low_sample_warning": low_sample,
            })

    return blocks


# ---------------------------------------------------------------------------
# Point slimming (to keep multi-match prompts stable)
# ---------------------------------------------------------------------------

def _slim_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only fields useful for grounding and timestamp citations (quality-first).

    This is used primarily for multi-match prompts to keep input stable.
    We retain a few extra coaching-relevant fields if they exist:
    - rally_length
    - serve_num_used (1st vs 2nd)
    - final_serve_dir (WIDE/T/BODY/UNKNOWN)
    - sides (deuce/adv)
    - pressure_point flag
    """
    slim: List[Dict[str, Any]] = []
    for p in (points or []):
        slim.append({
            "point_idx": p.get("point_idx"),
            "start_s": p.get("start_s"),
            "end_s": p.get("end_s"),
            "rally_summary": p.get("rally_summary"),
            "end_type": p.get("end_type"),
            "server": p.get("server"),
            "pt_won_by": p.get("pt_won_by"),
            # Extra coaching context (optional)
            "rally_length": p.get("rally_length"),
            "serve_num_used": p.get("serve_num_used"),
            "final_serve_dir": p.get("final_serve_dir"),
            "sides": p.get("sides"),
            "pressure_point": p.get("pressure_point"),
            # Key point flags (optional)
            "breakpoint": p.get("breakpoint"),
            "gamepoint": p.get("gamepoint"),
            "setpoint": p.get("setpoint"),
            "matchpoint": p.get("matchpoint"),
        })
    return slim

# ---------------------------------------------------------------------------
# Single match: insights + chat
# ---------------------------------------------------------------------------

@st.cache_data
def generate_insights(
    match_metadata: Dict[str, Any],
    evidence_packet: Dict[str, Any],
    candidate_points: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/insight_writer.md",
    max_output_tokens: int = 2600,
) -> Dict[str, Any]:
    client = OpenAI()
    model = model or _default_insight_model()
    system_prompt = _load_prompt(prompt_path)

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "MATCH_METADATA\n" + json.dumps(match_metadata, ensure_ascii=False) +
            "\n\nEVIDENCE_PACKET\n" + json.dumps(make_llm_evidence_view(evidence_packet), ensure_ascii=False) +
            "\n\nCANDIDATE_POINTS\n" + json.dumps(candidate_points, ensure_ascii=False) +
            "\n\nReturn ONLY valid JSON matching the schema."
        )}
    ]

    resp = client.responses.create(
        model=model,
        input=input_msgs,
        text={
            "format": {
                "type": "json_schema",
                "name": "match_insights",
                "schema": INSIGHTS_JSON_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )

    raw = _extract_response_text(resp)
    try:
        return _parse_json_or_raise(raw, model=model)
    except ValueError:
        # Retry once with a stronger repair instruction (quality-first)
        input_msgs_repair = list(input_msgs)
        input_msgs_repair.append({"role": "user", "content": _repair_instruction()})
        input_msgs_repair.append({"role": "user", "content": "Also, limit to at most 12 insights to ensure the JSON is complete."})
        input_msgs_repair.append({"role": "user", "content": "Also, limit to at most 15 insights to ensure the JSON is complete."})
        resp2 = client.responses.create(
            model=model,
            input=input_msgs_repair,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "match_insights",
                    "schema": INSIGHTS_JSON_SCHEMA,
                    "strict": True,
                }
            },
            max_output_tokens=max_output_tokens,
        )
        raw2 = _extract_response_text(resp2)
        return _parse_json_or_raise(raw2, model=model)


def answer_question(
    question: str,
    match_metadata: Dict[str, Any],
    evidence_packet: Dict[str, Any],
    insights_payload: Dict[str, Any],
    candidate_points: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/chat_answerer.md",
    max_output_tokens: int = 500,
) -> str:
    client = OpenAI()
    model = model or _default_chat_model()
    system_prompt = _load_prompt(prompt_path)

    slim_points = [
        {
            "point_idx": p.get("point_idx"),
            "start_s": p.get("start_s"),
            "end_s": p.get("end_s"),
            "rally_summary": p.get("rally_summary"),
            "end_type": p.get("end_type"),
        }
        for p in candidate_points
    ]

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "MATCH_METADATA\n" + json.dumps(match_metadata, ensure_ascii=False) +
            "\n\nEVIDENCE_PACKET\n" + json.dumps(make_llm_evidence_view(evidence_packet), ensure_ascii=False) +
            "\n\nINSIGHT_OBJECTS\n" + json.dumps(insights_payload, ensure_ascii=False) +
            "\n\nPOINTS\n" + json.dumps(slim_points, ensure_ascii=False) +
            "\n\nUSER_QUESTION\n" + question
        )}
    ]

    resp = client.responses.create(
        model=model,
        input=input_msgs,
        max_output_tokens=max_output_tokens,
    )

    return _extract_response_text(resp) or ""


# ---------------------------------------------------------------------------
# Multi-match: insights + chat
# ---------------------------------------------------------------------------

@st.cache_data
def generate_multi_match_insights(
    matches_payload: List[Dict[str, Any]],
    *,
    focus_player: Dict[str, Any],
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/multi_match_insight_writer.md",
    max_output_tokens: int = 1600,
) -> Dict[str, Any]:
    client = OpenAI()
    model = model or _default_insight_model()
    system_prompt = _load_prompt(prompt_path)

    matches_slim = []
    for m in matches_payload:
        matches_slim.append({
            "label": m["label"],
            "match_metadata": m["match_metadata"],
            "evidence_packet": make_llm_evidence_view(m["evidence_packet"]),
            "candidate_points": _slim_points(m["candidates"]),
        })

    # Precompute authoritative trend blocks so the LLM never has to compute
    # deltas or interpret directions itself (prevents math/direction mistakes).
    trend_blocks = _build_trend_blocks(matches_slim, focus_player)

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "FOCUS_PLAYER\n" + json.dumps(focus_player, ensure_ascii=False) +
            "\n\nMATCHES\n" + json.dumps(matches_slim, ensure_ascii=False) +
            "\n\nTREND_BLOCKS\n" + json.dumps(trend_blocks, ensure_ascii=False) +
            "\n\nReturn ONLY valid JSON matching the schema."
        )}
    ]

    resp = client.responses.create(
        model=model,
        input=input_msgs,
        text={
            "format": {
                "type": "json_schema",
                "name": "multi_match_insights",
                "schema": MULTI_INSIGHTS_JSON_SCHEMA,
                "strict": True,
            }
        },
        max_output_tokens=max_output_tokens,
    )

    raw = _extract_response_text(resp)
    try:
        return _parse_json_or_raise(raw, model=model)
    except ValueError:
        input_msgs_repair = list(input_msgs)
        input_msgs_repair.append({"role": "user", "content": _repair_instruction()})
        resp2 = client.responses.create(
            model=model,
            input=input_msgs_repair,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "multi_match_insights",
                    "schema": MULTI_INSIGHTS_JSON_SCHEMA,
                    "strict": True,
                }
            },
            max_output_tokens=max_output_tokens,
        )
        raw2 = _extract_response_text(resp2)
        return _parse_json_or_raise(raw2, model=model)


def answer_question_multi(
    question: str,
    matches_payload: List[Dict[str, Any]],
    multi_match_insights: Dict[str, Any],
    *,
    focus_player: Dict[str, Any],
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/multi_match_chat_answerer.md",
    max_output_tokens: int = 650,
) -> str:
    client = OpenAI()
    model = model or _default_chat_model()
    system_prompt = _load_prompt(prompt_path)

    # slim each match payload to reduce tokens but keep evidence + timestamps
    matches_slim = []
    for m in matches_payload:
        cand = m.get("candidates") or []
        slim_points = [
            {
                "point_idx": p.get("point_idx"),
                "start_s": p.get("start_s"),
                "end_s": p.get("end_s"),
                "rally_summary": p.get("rally_summary"),
                "end_type": p.get("end_type"),
            }
            for p in cand
        ]
        matches_slim.append({
            "label": m["label"],
            "match_metadata": m["match_metadata"],
            "evidence_packet": make_llm_evidence_view(m["evidence_packet"]),
            "insight_objects": m.get("insights"),
            "points": slim_points,
        })

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "FOCUS_PLAYER\n" + json.dumps(focus_player, ensure_ascii=False) +
            "\n\nMATCHES\n" + json.dumps(matches_slim, ensure_ascii=False) +
            "\n\nMULTI_MATCH_INSIGHTS\n" + json.dumps(multi_match_insights, ensure_ascii=False) +
            "\n\nUSER_QUESTION\n" + question
        )}
    ]

    resp = client.responses.create(
        model=model,
        input=input_msgs,
        max_output_tokens=max_output_tokens,
    )

    return _extract_response_text(resp) or ""
