from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from openai import OpenAI
from .schemas import INSIGHTS_JSON_SCHEMA

def _load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")

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
