from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_search_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_sqldb",
            "description": "Search structured SQL records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def _default_handle_sqldb_search_tool(tool_args: dict[str, Any], sql_search_fn) -> str:
    query = str(tool_args.get("query", "")).strip()
    max_results = int(tool_args.get("max_results", 10))
    if not query:
        return json.dumps({"error": "Empty query", "results": []})
    raw_results = sql_search_fn(query, k=max(1, min(max_results, 20)))
    results = [{"type": "text", "content": line} for line in raw_results.split("\n") if line.strip()]
    return json.dumps({"query": query, "returned": len(results), "results": results})


def _is_chat_openai_model(model: Any) -> bool:
    try:
        return isinstance(model, ChatOpenAI)
    except TypeError:
        # Some tests shim langchain_openai with non-type placeholders.
        return False


def sql_db_agent(
    state: dict[str, Any],
    model: Any,
    sql_search_fn,
    search_tool: dict[str, Any] | None = None,
    handle_tool_fn=None,
    stamp_fn=None,
) -> dict[str, Any]:
    search_tool = search_tool or _default_search_tool()
    handle_tool_fn = handle_tool_fn or _default_handle_sqldb_search_tool
    stamp_fn = stamp_fn or _default_stamp

    if "sql_db" not in state.get("active_agents", []):
        return {
            "sql_findings": "(not activated)",
            "messages": [AIMessage(content="[SQLDB] skipped")],
            "activity_log": [{
                "agent": "sql_db",
                "icon": "🗄️",
                "title": "SQL DB — skipped",
                "detail": "Router did not activate.",
                "ts": stamp_fn(),
            }],
            "current_agent": "sql_db",
        }

    if not _is_chat_openai_model(model):
        raw = sql_search_fn(state["query"], k=8)
        rows = [l for l in raw.split("\n") if l.strip()]
        keywords_str = ", ".join(state.get("keywords", [])) or "(general)"
        system = SystemMessage(content=(
            "You are a SQL Database Agent. Extract the most relevant structured information "
            "from the SQL query results, prioritizing content related to scoping keywords."
        ))
        resp = model.invoke([system, HumanMessage(content=f"Query: {state['query']}\nFocus keywords: {keywords_str}\n\nSQL results:\n{raw}")])

        return {
            "sql_findings": resp.content,
            "messages": [AIMessage(content=f"[SQLDB] {resp.content[:120]}…")],
            "activity_log": [{
                "agent": "sql_db",
                "icon": "🗄️",
                "title": "SQL / DB agent (direct search)",
                "detail": f"{len(rows)} result(s) found",
                "rows": rows[:12],
                "ts": stamp_fn(),
            }],
            "current_agent": "sql_db",
        }

    system_prompt = SystemMessage(content=(
        "You are a SQL Database Agent. You have access to a database of structured topics, "
        "relationships, and facts. Based on the user's query and scoping keywords, decide:\n"
        "1. Whether querying the SQL database would be helpful\n"
        "2. What search query to use (prioritize scoping keywords if present)\n"
        "3. How many results to retrieve (1-20)\n\n"
        "Use the search_sqldb tool if you think the SQL database has relevant structured information. "
        "Otherwise, respond explaining why a SQL search is not needed."
    ))

    keywords_str = ", ".join(state.get("keywords", [])) or "(none)"
    sub_q_str = "\n  ".join(state.get("sub_questions", [])[:3]) or "(none)"

    query_context = f"""User Query: {state['query']}

Scoping Keywords (prioritize in search): {keywords_str}

Research angles to address:
  {sub_q_str}

Decide whether and how to search the SQL database for structured facts."""

    messages = [system_prompt, HumanMessage(content=query_context)]

    try:
        response = model.invoke(messages, tools=[search_tool], tool_choice="auto")

        if hasattr(response, "tool_calls") and response.tool_calls:
            results_found = []
            tool_reasoning = []

            for tool_call in response.tool_calls:
                if tool_call.function.name == "search_sqldb":
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_result = handle_tool_fn(tool_args, sql_search_fn)
                        result_data = json.loads(tool_result)
                        if result_data.get("results"):
                            results_found.extend(result_data["results"])
                            tool_reasoning.append(
                                f"Queried for '{result_data.get('query')}' "
                                f"→ Found {result_data.get('returned')} results"
                            )
                    except Exception as e:
                        tool_reasoning.append(f"Tool error: {e}")

            if results_found:
                formatted_results = "\n".join([r["content"] for r in results_found])
                synthesis_system = SystemMessage(content=(
                    "Synthesise the SQL database results into structured research notes. "
                    "Preserve the topic, relationship, and fact distinctions."
                ))
                synthesis_resp = model.invoke([
                    synthesis_system,
                    HumanMessage(content=f"Query: {state['query']}\n\nDatabase results:\n{formatted_results}"),
                ])
                sql_findings = str(synthesis_resp.content)
            else:
                sql_findings = "(No relevant records found in SQL database)"

            return {
                "sql_findings": sql_findings,
                "messages": [AIMessage(content=f"[SQLDB] Searched {len(results_found)} result(s)")],
                "activity_log": [{
                    "agent": "sql_db",
                    "icon": "🗄️",
                    "title": "SQL / DB agent (tool-driven)",
                    "detail": f"LLM searched SQL: {'; '.join(tool_reasoning) or 'no results'}",
                    "rows": results_found[:12],
                    "ts": stamp_fn(),
                }],
                "current_agent": "sql_db",
            }

        llm_decision = response.content if hasattr(response, "content") else str(response)
        return {
            "sql_findings": f"(SQL search not needed: {str(llm_decision)[:200]})",
            "messages": [AIMessage(content="[SQLDB] Skipped search (LLM decision)")],
            "activity_log": [{
                "agent": "sql_db",
                "icon": "🗄️",
                "title": "SQL / DB agent (LLM decision)",
                "detail": f"LLM decided no SQL search needed: {str(llm_decision)[:100]}",
                "ts": stamp_fn(),
            }],
            "current_agent": "sql_db",
        }

    except Exception as e:
        return {
            "sql_findings": f"(SQL error: {str(e)[:100]})",
            "messages": [AIMessage(content=f"[SQLDB] Error: {str(e)[:50]}")],
            "activity_log": [{
                "agent": "sql_db",
                "icon": "🗄️",
                "title": "SQL / DB agent (error)",
                "detail": f"Tool execution failed: {str(e)[:100]}",
                "ts": stamp_fn(),
            }],
            "current_agent": "sql_db",
        }
