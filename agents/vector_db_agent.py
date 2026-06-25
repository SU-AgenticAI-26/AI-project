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
            "name": "search_vectordb",
            "description": "Search indexed documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "filter_source": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def _default_tool_handler(vdb: Any, tool_args: dict[str, Any]) -> str:
    query = str(tool_args.get("query", "")).strip()
    top_k = int(tool_args.get("top_k", 5))
    if not query:
        return json.dumps({"query": "", "returned": 0, "results": []})
    docs = vdb.search(query, k=max(1, min(top_k, 20)))
    results = []
    for doc in docs:
        results.append({
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content[:500],
            "metadata": {
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "indexed_at": doc.metadata.get("indexed_at", ""),
            },
        })
    return json.dumps({"query": query, "returned": len(results), "results": results})


def _is_chat_openai_model(model: Any) -> bool:
    try:
        return isinstance(model, ChatOpenAI)
    except TypeError:
        # Some tests shim langchain_openai with non-type placeholders.
        return False


def vector_db_agent(
    state: dict[str, Any],
    model: Any,
    vdb: Any,
    search_tool: dict[str, Any] | None = None,
    handle_tool_fn=None,
    stamp_fn=None,
) -> dict[str, Any]:
    search_tool = search_tool or _default_search_tool()
    handle_tool_fn = handle_tool_fn or _default_tool_handler
    stamp_fn = stamp_fn or _default_stamp

    if "vector_db" not in state.get("active_agents", []):
        return {
            "vector_findings": "(not activated)",
            "messages": [AIMessage(content="[VectorDB] skipped")],
            "activity_log": [{
                "agent": "vector_db",
                "icon": "🗂️",
                "title": "Vector DB — skipped",
                "detail": "Router did not activate.",
                "ts": stamp_fn(),
            }],
            "current_agent": "vector_db",
        }

    keywords_str = ", ".join(state.get("keywords", [])) or "(none)"
    sub_q_str = "\n  ".join(state.get("sub_questions", [])[:3]) or "(none)"

    if not _is_chat_openai_model(model):
        docs = vdb.search(state["query"], k=5)
        if not docs:
            raw_ctx = "(no documents indexed)"
            sources = []
        else:
            raw_ctx = "\n\n---\n".join(
                f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
            )
            sources = list({d.metadata.get("source", "?") for d in docs})

        system = SystemMessage(content=(
            "You are a Vector DB Search Agent. Synthesise the retrieved document chunks into "
            "structured research notes relevant to the query and scoping keywords."
        ))
        resp = model.invoke([system, HumanMessage(content=(
            f"Query: {state['query']}\n"
            f"Focus keywords: {keywords_str}\n"
            f"Research angles:\n  {sub_q_str}\n\n"
            f"Chunks:\n{raw_ctx}"
        ))])

        return {
            "vector_findings": resp.content,
            "messages": [AIMessage(content=f"[VectorDB] {resp.content[:120]}…")],
            "activity_log": [{
                "agent": "vector_db",
                "icon": "🗂️",
                "title": "Vector DB agent (direct search)",
                "detail": f"Retrieved {len(docs)} chunks from {len(sources)} source(s)",
                "ts": stamp_fn(),
            }],
            "current_agent": "vector_db",
        }

    def _synthesise_from_docs(docs_found: list[dict[str, Any]]) -> str:
        if not docs_found:
            return "(No relevant documents found in Vector DB)"
        formatted_docs = "\n\n---\n".join(
            f"[{d['source']}] {d['content']}" for d in docs_found[:6]
        )
        synthesis_resp = model.invoke([
            SystemMessage(content=(
                "Synthesise the retrieved VectorDB documents into structured research notes. "
                "Preserve source information and organise findings clearly."
            )),
            HumanMessage(content=(
                f"Query: {state['query']}\n"
                f"Focus keywords: {keywords_str}\n\n"
                f"Documents:\n{formatted_docs}"
            )),
        ])
        return str(synthesis_resp.content)

    def _fallback_docs(query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        docs = vdb.search(query_text, k=max(1, min(top_k, 20)))
        out: list[dict[str, Any]] = []
        for doc in docs:
            out.append({
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:500],
                "metadata": {
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "indexed_at": doc.metadata.get("indexed_at", ""),
                },
            })
        return out

    system_prompt = SystemMessage(content=(
        "You are a Vector DB Search Agent. You have access to a database of indexed documents, "
        "papers, and web search results. Based on the user's query and scoping context, decide:\n"
        "1. Whether searching the Vector DB would be helpful\n"
        "2. What search query to use (prioritize scoped keywords if present)\n"
        "3. How many results to retrieve (1-20)\n"
        "4. Whether to filter by a specific source type (web_search, arxiv, conference, etc.)\n\n"
        "Use the search_vectordb tool if you think the Vector DB has relevant information. "
        "Otherwise, respond explaining why a search is not needed."
    ))

    query_message = HumanMessage(content=(
        f"User Query: {state['query']}\n\n"
        f"Scoping keywords (use to refine search): {keywords_str}\n\n"
        f"Research angles to address:\n  {sub_q_str}\n\n"
        f"Decide whether and how to search Vector DB for relevant documents."
    ))

    try:
        response = model.invoke(
            [system_prompt, query_message],
            tools=[search_tool],
            tool_choice="auto",
        )

        if hasattr(response, "tool_calls") and response.tool_calls:
            docs_found = []
            tool_reasoning = []

            for tool_call in response.tool_calls:
                if tool_call.function.name == "search_vectordb":
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_result = handle_tool_fn(vdb, tool_args)
                        result_data = json.loads(tool_result)
                        if result_data.get("results"):
                            docs_found.extend(result_data["results"])
                            tool_reasoning.append(
                                f"Searched '{result_data.get('query')}' "
                                f"→ {result_data.get('returned')} results"
                            )
                    except Exception as e:
                        tool_reasoning.append(f"Tool error: {e}")

            if not docs_found:
                docs_found = _fallback_docs(state["query"], top_k=5)
                if docs_found:
                    tool_reasoning.append(
                        f"Fallback direct search on query -> {len(docs_found)} results"
                    )

            vector_findings = _synthesise_from_docs(docs_found)

            sources = list({d["source"] for d in docs_found})
            return {
                "vector_findings": vector_findings,
                "messages": [AIMessage(content=f"[VectorDB] {len(sources)} source(s), {len(docs_found)} docs")],
                "activity_log": [{
                    "agent": "vector_db",
                    "icon": "🗂️",
                    "title": "Vector DB agent (tool-driven)",
                    "detail": f"{'; '.join(tool_reasoning) or 'no results'}",
                    "docs_found": len(docs_found),
                    "sources": sources,
                    "ts": stamp_fn(),
                }],
                "current_agent": "vector_db",
            }

        llm_decision = getattr(response, "content", str(response))
        docs_found = _fallback_docs(state["query"], top_k=5)
        vector_findings = _synthesise_from_docs(docs_found)
        sources = list({d["source"] for d in docs_found})
        return {
            "vector_findings": vector_findings,
            "messages": [AIMessage(content=f"[VectorDB] fallback search {len(docs_found)} doc(s)")],
            "activity_log": [{
                "agent": "vector_db",
                "icon": "🗂️",
                "title": "Vector DB agent (fallback search)",
                "detail": (
                    f"LLM skipped tool call ({str(llm_decision)[:80]}). "
                    f"Fallback direct search returned {len(docs_found)} doc(s)."
                ),
                "docs_found": len(docs_found),
                "sources": sources,
                "ts": stamp_fn(),
            }],
            "current_agent": "vector_db",
        }

    except Exception as e:
        return {
            "vector_findings": f"(Vector DB error: {str(e)[:100]})",
            "messages": [AIMessage(content=f"[VectorDB] Error: {str(e)[:50]}")],
            "activity_log": [{
                "agent": "vector_db",
                "icon": "🗂️",
                "title": "Vector DB agent (error)",
                "detail": f"Tool execution failed: {str(e)[:100]}",
                "ts": stamp_fn(),
            }],
            "current_agent": "vector_db",
        }
