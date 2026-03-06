from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
import streamlit as st
from .schemas import INSIGHTS_JSON_SCHEMA, MULTI_INSIGHTS_JSON_SCHEMA

def _load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")

@st.cache_data
def generate_insights(
    match_metadata: Dict[str, Any],
    evidence_packet: Dict[str, Any],
    candidate_points: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/insight_writer.md",
    max_output_tokens: int = 1400,
) -> Dict[str, Any]:
    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = _load_prompt(prompt_path)

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "MATCH_METADATA\n" + json.dumps(match_metadata, ensure_ascii=False) +
            "\n\nEVIDENCE_PACKET\n" + json.dumps(evidence_packet, ensure_ascii=False) +
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
    return json.loads(resp.output_text)

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
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = _load_prompt(prompt_path)

    slim_points = [
        {"point_idx": p["point_idx"], "start_s": p["start_s"], "end_s": p["end_s"], "rally_summary": p["rally_summary"], "end_type": p["end_type"]}
        for p in candidate_points
    ]

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "MATCH_METADATA\n" + json.dumps(match_metadata, ensure_ascii=False) +
            "\n\nEVIDENCE_PACKET\n" + json.dumps(evidence_packet, ensure_ascii=False) +
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
    return resp.output_text


# ---------------------------------------------------------------------------
# Multi-match functions
# ---------------------------------------------------------------------------

@st.cache_data
def generate_multi_match_insights(
    matches: List[Dict[str, Any]],
    focus_player: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/multi_match_insight_writer.md",
    max_output_tokens: int = 2000,
) -> Dict[str, Any]:
    """Generate joint + trend insights across multiple matches.

    Each item in `matches` should be a dict with keys:
      - label: str  (e.g. "match_1", "match_2")
      - match_metadata: dict
      - evidence_packet: dict
      - candidates: list[dict]

    focus_player: optional dict with keys "id" and "name" to scope analysis to one player.
    """
    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = _load_prompt(prompt_path)

    matches_payload = []
    for m in matches:
        matches_payload.append({
            "label": m["label"],
            "match_metadata": m["match_metadata"],
            "evidence_packet": m["evidence_packet"],
            "candidate_points": m["candidates"],
        })

    user_content = ""
    if focus_player:
        user_content += "FOCUS_PLAYER\n" + json.dumps(focus_player, ensure_ascii=False) + "\n\n"
    user_content += "MATCHES\n" + json.dumps(matches_payload, ensure_ascii=False)
    user_content += "\n\nReturn ONLY valid JSON matching the schema."

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
    return json.loads(resp.output_text)


def answer_question_multi(
    question: str,
    matches: List[Dict[str, Any]],
    multi_insights: Dict[str, Any],
    focus_player: Optional[Dict[str, Any]] = None,
    *,
    model: Optional[str] = None,
    prompt_path: str | Path = "prompts/multi_match_chat_answerer.md",
    max_output_tokens: int = 700,
) -> str:
    """Answer a question using context from multiple matches.

    Each item in `matches` should be a dict with keys:
      - label: str
      - match_metadata: dict
      - evidence_packet: dict
      - insights: dict
      - candidates: list[dict]

    focus_player: optional dict with keys "id" and "name" to scope the answer to one player.
    """
    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    system_prompt = _load_prompt(prompt_path)

    matches_payload = []
    for m in matches:
        slim_points = [
            {
                "point_idx": p["point_idx"],
                "start_s": p["start_s"],
                "end_s": p["end_s"],
                "rally_summary": p["rally_summary"],
                "end_type": p["end_type"],
            }
            for p in (m.get("candidates") or [])
        ]
        matches_payload.append({
            "label": m["label"],
            "match_metadata": m["match_metadata"],
            "evidence_packet": m["evidence_packet"],
            "insight_objects": m.get("insights") or {},
            "points": slim_points,
        })

    user_content = ""
    if focus_player:
        user_content += "FOCUS_PLAYER\n" + json.dumps(focus_player, ensure_ascii=False) + "\n\n"
    user_content += (
        "MATCHES\n" + json.dumps(matches_payload, ensure_ascii=False) +
        "\n\nMULTI_MATCH_INSIGHTS\n" + json.dumps(multi_insights, ensure_ascii=False) +
        "\n\nUSER_QUESTION\n" + question
    )

    input_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    resp = client.responses.create(
        model=model,
        input=input_msgs,
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text
