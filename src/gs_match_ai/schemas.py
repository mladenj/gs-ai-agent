from __future__ import annotations

INSIGHTS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_summary": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "insights": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "coaching_tip": {"type": "string"},
                    "priority": {"type": "number"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "supporting_points": {"type": "array", "items": {"type": "integer"}, "minItems": 1}
                },
                "required": ["topic","title","summary","coaching_tip","priority","evidence_refs","supporting_points"],
                "additionalProperties": False
            }
        },
    },
    "required": ["entry_summary", "insights"],
    "additionalProperties": False
}

MULTI_INSIGHTS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "entry_summary": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "insights": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "coaching_tip": {"type": "string"},
                    "priority": {"type": "number"},
                    "match_scope": {"type": "string"},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "supporting_points": {"type": "array", "items": {"type": "integer"}, "minItems": 0}
                },
                "required": ["topic","title","summary","coaching_tip","priority","match_scope","evidence_refs","supporting_points"],
                "additionalProperties": False
            }
        },
    },
    "required": ["entry_summary", "insights"],
    "additionalProperties": False
}
