from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:
    from conference_paper_search import SEARCH_PAPERS_TOOL, handle_conference_paper_tool_call
    HAS_CONFERENCE_SEARCH = True
except Exception:
    HAS_CONFERENCE_SEARCH = False
    SEARCH_PAPERS_TOOL = None


def _default_stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def web_agent(state: dict[str, Any], model: Any, vdb: Any, stamp_fn=None) -> dict[str, Any]:
    stamp_fn = stamp_fn or _default_stamp
    if "web" not in state.get("active_agents", []):
        return {
            "web_findings": "(not activated)",
            "messages": [AIMessage(content="[Web] skipped")],
            "activity_log": [{
                "agent": "web",
                "icon": "🌐",
                "title": "Web agent — skipped",
                "detail": "Router did not activate.",
                "ts": stamp_fn(),
            }],
            "current_agent": "web",
        }

    query = state["query"]
    results_text = []
    errors = []
    indexed = 0
    sources_used = []

    if HAS_CONFERENCE_SEARCH and SEARCH_PAPERS_TOOL is not None:
        try:
            query_lower = query.lower()
            conference_keywords = {"neurips", "icml", "iclr", "acl", "emnlp", "naacl", "eacl", "conference", "paper", "arxiv"}
            mentions_conference = any(kw in query_lower for kw in conference_keywords)

            if mentions_conference and isinstance(model, ChatOpenAI):
                system_prompt = SystemMessage(content=(
                    "You are a research assistant. If the user is looking for papers from ML/NLP conferences "
                    "(NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, EACL), you should use the search_conference_papers tool "
                    "to find the most relevant papers. Determine which conferences, years, and keywords are most relevant "
                    "from the user's query, then call the tool accordingly."
                ))
                messages = [system_prompt, HumanMessage(content=f"Search for papers related to: {query}")]
                response = model.invoke(messages, tools=[SEARCH_PAPERS_TOOL], tool_choice="auto")

                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call.function.name == "search_conference_papers":
                            try:
                                tool_args = json.loads(tool_call.function.arguments)
                                tool_result = handle_conference_paper_tool_call(tool_args)
                                result_data = json.loads(tool_result)
                                papers = result_data.get("papers", [])
                                if papers:
                                    sources_used.append(f"Conference Papers ({len(papers)})")
                                    for paper in papers[:5]:
                                        title = paper.get("title", "")
                                        conference = paper.get("conference", "")
                                        year = paper.get("year", "")
                                        authors = paper.get("authors", "")
                                        results_text.append(
                                            f"[{conference} {year}] {title}\nAuthors: {authors}\nURL: {paper.get('url', '')}"
                                        )
                                        vdb.add_text(
                                            f"Title: {title}\nConference: {conference}\nYear: {year}\n"
                                            f"Authors: {authors}\nAbstract: {paper.get('abstract', '')}",
                                            {"source": f"{conference}_{year}", "title": title, "url": paper.get("url", "")},
                                        )
                                        indexed += 1
                            except Exception as e:
                                results_text.append(f"[Conference Paper Search error] {e}")
        except Exception:
            pass

    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": 5, "mailto": "research@example.com"},
            timeout=10,
        )
        if r.ok:
            for item in r.json().get("results", []):
                title = item.get("title", "No title")
                year = item.get("publication_year", "")
                authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])[:3]]
                results_text.append(f"[OpenAlex] {title} ({year}) — {', '.join(authors)}")
                vdb.add_text(f"Title: {title}\nAuthors: {', '.join(authors)}\nYear: {year}", {"source": "openalex", "title": title})
                indexed += 1
            sources_used.append("OpenAlex")
    except Exception as e:
        errors.append(f"OpenAlex: {e}")

    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": 5, "mailto": "research@example.com"},
            timeout=10,
        )
        if r.ok:
            for item in r.json().get("message", {}).get("items", []):
                title = (item.get("title") or ["No title"])[0]
                doi = item.get("DOI", "")
                authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])[:3]]
                results_text.append(f"[Crossref] {title} — {', '.join(authors)} | doi:{doi}")
                vdb.add_text(f"Title: {title}\nAuthors: {', '.join(authors)}\nDOI: {doi}", {"source": "crossref", "title": title})
                indexed += 1
            sources_used.append("Crossref")
    except Exception as e:
        errors.append(f"Crossref: {e}")

    try:
        ss_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {"x-api-key": ss_key} if ss_key else {}
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": query, "limit": 5, "fields": "title,year,citationCount,authors"},
            headers=headers,
            timeout=10,
        )
        if r.ok:
            for paper in r.json().get("data", []):
                title = paper.get("title", "No title")
                year = paper.get("year", "")
                cites = paper.get("citationCount", 0)
                authors = [a.get("name", "") for a in (paper.get("authors") or [])[:3]]
                results_text.append(f"[Semantic Scholar] {title} ({year}) — {', '.join(authors)} — {cites} citations")
                vdb.add_text(
                    f"Title: {title}\nAuthors: {', '.join(authors)}\nYear: {year}\nCitations: {cites}",
                    {"source": "semantic_scholar", "title": title},
                )
                indexed += 1
            sources_used.append("Semantic Scholar")
    except Exception as e:
        errors.append(f"Semantic Scholar: {e}")

    try:
        encoded = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results=5&sortBy=relevance"
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml = resp.read().decode("utf-8")

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        entries = root.findall("atom:entry", ns)
        for entry in entries:
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip()[:400]
            eid = (entry.findtext("atom:id", "", ns) or "").strip()
            authors = [a.findtext("atom:name", "", ns) for a in entry.findall("atom:author", ns)]
            results_text.append(f"[arXiv] {title} — {', '.join(authors[:3])}\n{eid}")
            vdb.add_text(
                f"Title: {title}\nAuthors: {', '.join(authors)}\nAbstract: {summary}",
                {"source": "arXiv", "title": title, "url": eid, "indexed_at": stamp_fn()},
            )
            indexed += 1
        sources_used.append("arXiv")
    except Exception as e:
        errors.append(f"arXiv: {e}")

    if results_text:
        combined = "\n\n---\n".join(results_text)
        if errors:
            combined += f"\n\n(Note: the following sources failed and returned no results: {'; '.join(errors)})"
    elif errors:
        combined = f"(All sources failed to return results. Errors: {'; '.join(errors)})"
    else:
        combined = "(no results)"

    system = SystemMessage(content=(
        "You are a Web Research Agent. Summarise the scholarly search results below into "
        "structured research notes relevant to the query. Cite the source for each finding "
        "(Conference Papers, OpenAlex, Crossref, Semantic Scholar, or arXiv)."
    ))
    resp = model.invoke([system, HumanMessage(content=f"Query: {query}\n\nResults:\n{combined}")])

    return {
        "web_findings": resp.content,
        "messages": [AIMessage(content=f"[Web] {resp.content[:120]}…")],
        "activity_log": [{
            "agent": "web",
            "icon": "🌐",
            "title": "Web / API agent",
            "detail": f"Sources queried: {', '.join(sources_used) or 'none'} — {indexed} papers indexed into VectorDB",
            "ts": stamp_fn(),
        }],
        "current_agent": "web",
    }
