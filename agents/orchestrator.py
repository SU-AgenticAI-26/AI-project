"""
agents/orchestrator.py — OrchestratorAgent

Coordinates specialist agents through a ReAct loop (max 15 turns).
At each turn it receives a structured state summary, produces visible
reasoning text, then selects a tool call. Routing is decided at runtime
by the LLM — not by predetermined graph edges.

Supports all providers via the LLMProvider abstraction:
  - Anthropic: native tool_use API
  - OpenAI / Azure / OpenAI-compatible: native function calling
  - Gemini: FunctionDeclaration
  - Ollama / llama.cpp: text-based ReAct fallback (THOUGHT/ACTION/ACTION_INPUT)

Adding a new agent
──────────────────
1. Create agents/your_agent.py — set tool_name, tool_description, tool_schema
   on the class (see agents/base.py for the protocol).
2. Import it below and add ("your_agent_key", YourAgentClass) to AGENT_REGISTRY
   at the correct pipeline position.
3. Add "your_agent_key" to AGENT_NAMES in llm/registry.py.
4. Add your output field(s) to state.py.
5. Add one import to agents/__init__.py.
TOOL_DEFINITIONS, _make_agents(), and the route map are auto-built from
AGENT_REGISTRY — no further changes to this file are needed.
"""

import json
from typing import Callable

from llm.base import LLMProvider, ToolCall
from llm.registry import AgentConfig, build_config_from_env
from message_bus import Message, MessageBus

# ── Agent registry ─────────────────────────────────────────────────────────────
# To add a new agent: import it and append ("state_key", AgentClass) here.
# The order defines the pipeline sequence shown to the Orchestrator LLM.
from agents.scoping            import ScopingAgent
from agents.search_reading     import SearchReadingAgent
from agents.reading_extraction import ReadingExtractionAgent
from agents.synthesis_planning import SynthesisPlanningAgent
from agents.validation         import ValidationAgent

AGENT_REGISTRY: list = [
    ("scoping_agent",            ScopingAgent),
    ("search_reading_agent",     SearchReadingAgent),
    ("reading_extraction_agent", ReadingExtractionAgent),
    ("synthesis_planning_agent", SynthesisPlanningAgent),
    ("validation_agent",         ValidationAgent),
]

# ── Auto-built from registry ───────────────────────────────────────────────────

_FINISH_TOOL = {
    "name": "finish",
    "description": (
        "Mark the pipeline complete and return results. Call when "
        "validation approves, or after 2 validation iterations."
    ),
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

TOOL_DEFINITIONS = [
    {
        "name":         agent_cls.tool_name,
        "description":  agent_cls.tool_description,
        "input_schema": agent_cls.tool_schema,
    }
    for _, agent_cls in AGENT_REGISTRY
    if agent_cls.tool_name
] + [_FINISH_TOOL]

# tool_name → state_key (used by _execute_tool)
_ROUTE_MAP: dict = {
    agent_cls.tool_name: state_key
    for state_key, agent_cls in AGENT_REGISTRY
    if agent_cls.tool_name
}


def _make_agents(agent_config: AgentConfig) -> dict:
    return {
        state_key: agent_cls(agent_config[state_key])
        for state_key, agent_cls in AGENT_REGISTRY
    }


MAX_TURNS = 15

SYSTEM_PROMPT = """\
You are the Orchestrator of a multi-agent academic research system. You coordinate
specialist agents by sending them task messages through a shared message bus.
Each agent is an independent reasoning entity with its own LLM instance.

YOUR AGENTS:
- ScopingAgent             — decomposes the research query into 3-5 sub-questions
- SearchReadingAgent       — selects sources from a 7-source registry (domain-aware),
                             retrieves, scores, deduplicates, and summarises papers
- ReadingExtractionAgent   — converts retrieved papers into structured 6-field records
                             (research_problem, methodology, findings, limitations,
                             future_work, key_claims) with provenance metadata
- SynthesisPlanningAgent   — produces a thematic literature review AND a research
                             plan; reads its own feedback from ValidationAgent
- ValidationAgent          — evaluates citation accuracy, coherence, and
                             completeness; sends directed feedback to
                             SynthesisPlanningAgent when issues are found

KEY MULTI-AGENT BEHAVIOUR:
When you call send_to_validation_agent, ValidationAgent sends a feedback message
DIRECTLY to SynthesisPlanningAgent on the bus. When you then call
send_to_synthesis_planning_agent again, that agent automatically reads its feedback,
produces an acknowledgement, and revises its outputs. The Orchestrator observes
both the validation result and the ack in the state summary.

STANDARD WORKFLOW:
1. send_to_scoping_agent                  — always first
2. send_to_search_reading_agent           — always second; searches all sub-questions
3. send_to_reading_extraction_agent       — always third; structures the retrieved papers
4. send_to_synthesis_planning_agent       — once extraction is complete
5. send_to_validation_agent              — evaluate outputs
6a. If APPROVED → finish
6b. If NOT APPROVED → ValidationAgent has ALREADY sent feedback to
    SynthesisPlanningAgent. Call send_to_synthesis_planning_agent (it reads
    its own feedback and revises). Then send_to_validation_agent again.
    After 2 validation iterations → always finish regardless.
7. If synthesis flagged uncovered sub-questions AND corpus is small (<8 papers):
   call send_to_search_reading_agent with targeted_query, then
   send_to_reading_extraction_agent again before synthesis.
8. finish — always the final call.

Read the state summary carefully before each decision."""


# ── State summary ─────────────────────────────────────────────────────────────

def _build_state_summary(state: dict, bus: MessageBus) -> str:
    sub_questions = state.get("sub_questions", [])
    all_papers    = state.get("all_papers", [])
    summaries     = state.get("paper_summaries", [])
    synthesis     = state.get("synthesis", "")
    plan          = state.get("research_plan", "")
    val_result    = state.get("validation_result", {})
    val_iters     = state.get("validation_iterations", 0)

    # Per-sub-question coverage estimate
    coverage = []
    for i, sq in enumerate(sub_questions, 1):
        keywords = [w.lower() for w in sq.split() if len(w) > 4][:5]
        count    = sum(
            1 for p in all_papers
            if any(kw in (p.get("title", "") + " " + p.get("abstract", "")).lower()
                   for kw in keywords)
        )
        coverage.append(f"  Sub-Q {i}: ~{count} papers — {sq[:55]}")

    # Source breakdown
    src_counts: dict = {}
    for p in all_papers:
        src = p.get("source", "?")
        src_counts[src] = src_counts.get(src, 0) + 1

    # Recent bus messages
    recent_msgs = [f"  {m.summary()}" for m in bus.all()[-6:]]

    # Pending feedback
    pending_fb = bus.feedback_for("synthesis_planning_agent")

    extracted     = state.get("extracted_papers", [])
    n_full_text   = sum(1 for p in extracted
                        if p.get("provenance", {}).get("text_source") == "full_text")
    n_abstract    = sum(1 for p in extracted
                        if p.get("provenance", {}).get("text_source") == "abstract_only")

    lines = [
        "══════════════════ STATE ══════════════════",
        f"Query: {state.get('query', '')}",
        f"Sub-questions: {len(sub_questions)}",
        *[f"  {i}. {q}" for i, q in enumerate(sub_questions, 1)],
        f"Sources selected: {state.get('sources_selected') or '(not yet)'}",
        f"Corpus: {len(all_papers)} papers  |  Summarised: {len(summaries)}",
        f"Extracted: {len(extracted)} papers"
        + (f" ({n_full_text} full-text, {n_abstract} abstract-only)" if extracted else " (not yet)"),
        f"By source: {src_counts or '(none yet)'}",
        *coverage,
        f"Synthesis: {'✓' if synthesis else '✗'}  |  Plan: {'✓' if plan else '✗'}  |  Validation iterations: {val_iters}",
    ]

    if val_result:
        approved = val_result.get("approved", False)
        issues   = val_result.get("issues", [])
        lines.append(
            f"Last validation: {'APPROVED' if approved else f'NOT APPROVED — {len(issues)} issue(s)'}"
        )
        for iss in issues[:4]:
            lines.append(f"  • {iss}")

    lines += ["Recent bus:"] + recent_msgs

    if pending_fb:
        lines.append(
            f"⚠ Feedback waiting for synthesis_planning_agent: "
            f"{len(pending_fb)} message(s)"
        )

    lines.append("═══════════════════════════════════════════")
    return "\n".join(lines)


# ── Agent factory ─────────────────────────────────────────────────────────────

def _make_agents(agent_config: AgentConfig) -> dict:
    return {
        "scoping_agent":            ScopingAgent(agent_config["scoping_agent"]),
        "search_reading_agent":     SearchReadingAgent(agent_config["search_reading_agent"]),
        "synthesis_planning_agent": SynthesisPlanningAgent(agent_config["synthesis_planning_agent"]),
        "validation_agent":         ValidationAgent(agent_config["validation_agent"]),
    }


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict, state: dict, bus: MessageBus,
                  agents: dict, callback: Callable) -> str:
    if name == "finish":
        state["is_complete"] = True
        bus.send(Message(sender="orchestrator", recipient="bus",
                         msg_type="result", content={"status": "complete"}))
        callback("tool_result", {"tool": "finish", "agent": "", "result": "Pipeline complete ✓"})
        return json.dumps({"status": "complete"})

    agent_key = _ROUTE_MAP.get(name)
    if not agent_key:
        return json.dumps({"error": f"Unknown tool: {name}"})

    content  = {**inputs, "query": state.get("query", "")}
    task_msg = Message(sender="orchestrator", recipient=agent_key,
                       msg_type="task", content=content)
    bus.send(task_msg)
    callback("tool_call", {"name": name, "inputs": inputs, "agent": agent_key})

    try:
        reply = agents[agent_key].run(task_msg, state, bus)
    except Exception as exc:
        err = f"Agent '{agent_key}' error: {exc}"
        callback("error", err)
        return json.dumps({"error": err})

    result_summary = reply.content.get("summary", str(reply.content)[:120])
    callback("tool_result", {"tool": name, "agent": agent_key, "result": result_summary})
    return json.dumps(reply.content)


# ── Provider-aware conversation management ────────────────────────────────────

def _is_anthropic(provider: LLMProvider) -> bool:
    return provider.__class__.__name__ == "AnthropicProvider"


def _append_turn(messages: list, raw_response, reasoning: str,
                 tool_calls: list[ToolCall], tool_results: list[dict],
                 provider: LLMProvider, state_summary: str) -> list:
    if _is_anthropic(provider) and raw_response is not None:
        messages.append({"role": "assistant", "content": raw_response.content})
        user_content = [
            {"type": "tool_result", "tool_use_id": tr["tool_use_id"],
             "content": tr["content"]}
            for tr in tool_results
        ] + [{"type": "text", "text": f"Updated state:\n\n{state_summary}"}]
        messages.append({"role": "user", "content": user_content})
    else:
        if reasoning:
            messages.append({"role": "assistant", "content": reasoning})
        obs = "\n\n".join(
            f"OBSERVATION [{tr['tool_use_id']}]: {tr['content']}"
            for tr in tool_results
        )
        user_text = (obs + "\n\n" if obs else "") + f"Updated state:\n\n{state_summary}"
        messages.append({"role": "user", "content": user_text})
    return messages


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_orchestrator(state: dict, callback: Callable,
                     agent_config: AgentConfig = None) -> dict:
    if agent_config is None:
        agent_config = build_config_from_env()

    provider = agent_config["orchestrator"]
    agents   = _make_agents(agent_config)
    bus      = MessageBus()
    state.setdefault("bus", bus)
    state["provider_summary"] = agent_config.summary()

    callback("status", f"Orchestrator using: {provider.name}")

    prior_ctx = state.get("prior_context", "")
    initial   = (
        f'Begin the research pipeline for this query:\n\n"{state["query"]}"\n\n'
        + (prior_ctx + "\n\n" if prior_ctx else "")
        + _build_state_summary(state, bus)
    )
    messages  = [{"role": "user", "content": initial}]

    for turn in range(MAX_TURNS):
        if state.get("is_complete"):
            break

        callback("status", f"Orchestrator turn {turn + 1}/{MAX_TURNS} [{provider.name}]")

        raw_response   = None
        reasoning_text = ""
        tool_calls: list[ToolCall] = []

        try:
            if _is_anthropic(provider):
                raw_response = provider.raw_response_content(
                    SYSTEM_PROMPT, messages, TOOL_DEFINITIONS, max_tokens=2000
                )
                for block in raw_response.content:
                    if not hasattr(block, "type"):
                        continue
                    if block.type == "text":
                        reasoning_text += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(
                            ToolCall(id=block.id, name=block.name, input=block.input or {})
                        )
                stop_reason = raw_response.stop_reason
            else:
                reasoning_text, tool_calls = provider.complete_with_tools(
                    SYSTEM_PROMPT, messages, TOOL_DEFINITIONS, max_tokens=2000
                )
                stop_reason = "tool_use" if tool_calls else "end_turn"

        except Exception as exc:
            callback("error", f"API error turn {turn + 1}: {exc}")
            state.setdefault("status_log", []).append(f"[Orch T{turn+1}] API error: {exc}")
            break

        if reasoning_text.strip():
            callback("reasoning", reasoning_text.strip())
            state.setdefault("orchestrator_trace", []).append(
                {"turn": turn + 1, "type": "reasoning", "content": reasoning_text.strip()}
            )

        if stop_reason == "end_turn" and not tool_calls:
            callback("status", "Orchestrator finished (no further tool calls)")
            break

        tool_results = []
        for tc in tool_calls:
            state.setdefault("status_log", []).append(
                f"[Orch T{turn+1}] → {tc.name}"
                + (f" query='{tc.input.get('targeted_query','')}'"
                   if "targeted_query" in tc.input else "")
            )
            state.setdefault("orchestrator_trace", []).append(
                {"turn": turn + 1, "type": "tool_call",
                 "tool": tc.name, "inputs": tc.input}
            )
            result_json = _execute_tool(tc.name, tc.input, state, bus, agents, callback)
            tool_results.append({"tool_use_id": tc.id, "content": result_json})

        updated_summary = _build_state_summary(state, bus)
        messages = _append_turn(
            messages, raw_response, reasoning_text,
            tool_calls, tool_results, provider, updated_summary
        )

        if state.get("is_complete"):
            break

    if not state.get("is_complete"):
        state["is_complete"] = True
        callback("status", "Turn limit reached — returning best available output")

    state["message_log"]  = bus.log_lines()
    state["bus_messages"] = [
        {
            "id":         m.id,
            "sender":     m.sender,
            "recipient":  m.recipient,
            "msg_type":   m.msg_type,
            "content":    m.content,
            "timestamp":  m.timestamp,
            "in_reply_to": m.in_reply_to,
            "summary":    m.summary(),
        }
        for m in bus.all()
    ]
    return state
