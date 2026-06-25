from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


SQL_TRIGGER_PATTERNS = [
    "what are the main",
    "what approaches",
    "what mechanisms",
    "approaches",
    "mechanisms",
    "challenges",
    "list the",
    "compare",
    "how many",
]

_SQL_TRIGGER_REGEXES = [
    # q2-style categorical prompts
    re.compile(r"\bwhat\s+are\s+(?:the\s+)?main\b"),
    re.compile(r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?:main\s+)?approaches\b"),
    re.compile(r"\bapproaches\s+and\s+challenges\b"),
    # q4-style mechanism prompts
    re.compile(r"\bwhat\s+(?:collaboration\s+)?mechanisms\b"),
    re.compile(r"\bcollaboration\s+mechanisms\b"),
    # generic enumeration / structured-fact prompts
    re.compile(r"\bmechanisms?\b"),
    re.compile(r"\bchallenges?\b"),
]


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _needs_sql_for_query(query: str) -> bool:
    q = (query or "").lower()
    if any(pattern in q for pattern in SQL_TRIGGER_PATTERNS):
        return True
    return any(rx.search(q) is not None for rx in _SQL_TRIGGER_REGEXES)


def router_agent(state: dict[str, Any], model: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    critique_context = ""
    if state.get("_needs_more"):
        critique_context = (
            f"\n\nPREVIOUS ATTEMPT FEEDBACK:\n{state.get('critique', 'Insufficient coverage detected.')}"
            "\nAdjust source selection strategy to address gaps."
        )

    keywords_str = ", ".join(state.get("keywords", [])[:8]) or "(none extracted)"
    sub_questions_count = len(state.get("sub_questions", []))

    system = SystemMessage(content=(
        "You are a Router Agent. Given a user query and refined scoping context, decide which search agents to activate.\n"
        "Available: 'vector_db' (semantic doc search), 'sql_db' (structured facts/topics), "
        "'web' (live scholarly search — use when query needs recent papers or external knowledge).\n"
        "Routing examples:\n"
        "- Query: 'What are the main approaches and challenges in federated learning for healthcare?' "
        "-> include 'sql_db' for categorical/structured coverage.\n"
        "- Query: 'What collaboration mechanisms are used in multi-agent LLM systems?' "
        "-> include 'sql_db' for mechanism enumeration.\n"
        "Return ONLY JSON: {\"agents\": [...], \"reasoning\": \"one sentence\"}. No other text."
    ))

    query_context = f"""Query: {state['query']}

Scoping Context:
- Key themes/keywords to prioritize: {keywords_str}
- Research angles: {sub_questions_count} sub-questions identified

Decide which sources would best address these angles.{critique_context}"""

    resp = model.invoke([system, HumanMessage(content=query_context)])
    raw = str(resp.content).strip().lstrip("```json").rstrip("```").strip()
    try:
        parsed = json.loads(raw)
        agents = parsed.get("agents", ["vector_db", "sql_db"])
        reason = parsed.get("reasoning", "")
    except Exception:
        agents = ["vector_db", "sql_db"]
        reason = "defaulted"

    if _needs_sql_for_query(state.get("query", "")) and "sql_db" not in agents:
        agents.append("sql_db")
        if reason:
            reason = f"{reason}; sql_db forced by SQL trigger pattern"
        else:
            reason = "sql_db forced by SQL trigger pattern"

    return {
        "active_agents": agents,
        "router_reasoning": reason,
        "messages": [AIMessage(content=f"[Router] {reason} → {agents}")],
        "activity_log": [{
            "agent": "router",
            "icon": "🔀",
            "title": "Router decided",
            "detail": f"Activating: {', '.join(agents)} — {reason}",
            "ts": stamp_fn(),
        }],
        "current_agent": "router",
    }
