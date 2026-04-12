"""
Conference Paper Search Tool
─────────────────────────────

Searches academic papers across major ML/NLP conferences:
  - OpenReview      → NeurIPS, ICML, ICLR (requires free OpenReview credentials)
  - ACL Anthology   → ACL, EMNLP, NAACL, EACL, COLING (no auth needed)

Install:
    pip install openreview-py acl-anthology

Set OpenReview credentials via environment variables:
    export OPENREVIEW_USERNAME="your@email.com"
    export OPENREVIEW_PASSWORD="yourpassword"

Usage:
    from conference_paper_search import search_conference_papers, SEARCH_PAPERS_TOOL
    
    results = search_conference_papers(
        keywords=["diffusion models"],
        years=[2023, 2024],
        conferences=["NeurIPS", "ICML"],
    )
    print(results)
"""

import json
import os
import re
import time
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# OPENAI TOOL DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_conference_papers",
        "description": (
            "Search for academic papers by keywords across top ML and NLP conferences. "
            "Supports OpenReview venues (NeurIPS, ICML, ICLR) and ACL Anthology venues "
            "(ACL, EMNLP, NAACL, EACL, COLING). Returns a list of matching papers with "
            "title, abstract, authors, keywords, and URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of keyword strings to search for (OR logic). E.g. [\"diffusion\", \"score matching\"]."
                },
                "years": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of publication years to include. E.g. [2022, 2023, 2024]."
                },
                "conferences": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "NAACL", "EACL"]
                    },
                    "description": "List of conferences to search. OpenReview: NeurIPS, ICML, ICLR. ACL Anthology: ACL, EMNLP, NAACL, EACL."
                },
                "search_in": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["title", "abstract"]},
                    "description": "Which fields to search in. Defaults to [\"title\", \"abstract\"]."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return per conference. Defaults to 10."
                },
            },
            "required": ["keywords", "years", "conferences"],
            "additionalProperties": False,
        },
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# OPENREVIEW CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

OR_USERNAME = os.environ.get("OPENREVIEW_USERNAME", "mchoudhu@g.syr.edu")
OR_PASSWORD = os.environ.get("OPENREVIEW_PASSWORD", "Walantly!Aa12")


OR_VENUES = {
    "NeurIPS": {
        2024: {"id": "NeurIPS.cc/2024/Conference", "api": 2},
        2023: {"id": "NeurIPS.cc/2023/Conference", "api": 2},
        2022: {"id": "NeurIPS.cc/2022/Conference", "api": 1},
        2021: {"id": "NeurIPS.cc/2021/Conference", "api": 1},
    },
    "ICML": {
        2024: {"id": "ICML.cc/2024/Conference", "api": 2},
        2023: {"id": "ICML.cc/2023/Conference", "api": 2},
        2022: {"id": "ICML.cc/2022/Conference", "api": 1},
        2021: {"id": "ICML.cc/2021/Conference", "api": 1},
    },
    "ICLR": {
        2024: {"id": "ICLR.cc/2024/Conference", "api": 2},
        2023: {"id": "ICLR.cc/2023/Conference", "api": 2},
        2022: {"id": "ICLR.cc/2022/Conference", "api": 1},
        2021: {"id": "ICLR.cc/2021/Conference", "api": 1},
    },
}

OPENREVIEW_CONFERENCES = {"NeurIPS", "ICML", "ICLR"}
ACL_CONFERENCES = {"ACL", "EMNLP", "NAACL", "EACL"}


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def make_pattern(keywords):
    """Create a regex pattern to match any of the keywords (case-insensitive)."""
    return re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)


def matches(text, pattern):
    """Check if pattern matches text."""
    return bool(pattern.search(text or ""))


def get_field(content, field):
    """Extract field value from content dict, handling nested structures."""
    val = content.get(field, "")
    if isinstance(val, dict):
        val = val.get("value", "")
    if isinstance(val, list):
        val = ", ".join(str(v) for v in val)
    return str(val).strip()


# ──────────────────────────────────────────────────────────────────────────────
# OPENREVIEW SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def get_or_clients():
    """Login to OpenReview API (both v1 and v2)."""
    try:
        import openreview
    except ImportError:
        return None, None

    if not OR_USERNAME or not OR_PASSWORD:
        print("  [OpenReview] Credentials not set. Skipping.")
        return None, None

    try:
        print("  [OpenReview] Logging in...")
        v2 = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=OR_USERNAME,
            password=OR_PASSWORD,
        )
        v1 = openreview.Client(
            baseurl="https://api.openreview.net",
            username=OR_USERNAME,
            password=OR_PASSWORD,
        )
        print("  [OpenReview] Logged in")
        return v1, v2
    except Exception as e:
        print(f"  [OpenReview] Login failed: {e}")
        return None, None


def fetch_api2(client, venue_id):
    """Fetch papers using OpenReview API v2."""
    if client is None:
        return []
    try:
        print(f"    [API2] {venue_id}...")
        papers = client.get_all_notes(content={"venueid": venue_id})
        print(f"    -> {len(papers)} papers")
        return papers
    except Exception as e:
        print(f"    [API2] Error: {e}")
        return []


def fetch_api1(client, venue_id):
    """Fetch accepted papers using OpenReview API v1."""
    if client is None:
        return []
    try:
        print(f"    [API1] {venue_id}...")
        try:
            subs = client.get_all_notes(
                invitation=f"{venue_id}/-/Blind_Submission", details="directReplies"
            )
        except Exception:
            subs = client.get_all_notes(
                invitation=f"{venue_id}/-/Submission", details="directReplies"
            )

        accepted = []
        for sub in subs:
            for r in sub.details.get("directReplies", []):
                if r.get("invitation", "").endswith("Decision"):
                    if "Accept" in str(r.get("content", {}).get("decision", "")):
                        accepted.append(sub)
                        break
        print(f"    -> {len(accepted)} accepted")
        return accepted
    except Exception as e:
        print(f"    [API1] Error: {e}")
        return []


def search_openreview(conferences, years, keywords, search_in):
    """Search OpenReview venues (NeurIPS, ICML, ICLR)."""
    results = []
    pattern = make_pattern(keywords)

    v1, v2 = get_or_clients()
    if v1 is None and v2 is None:
        return results

    for conf in conferences:
        for year in years:
            info = OR_VENUES.get(conf, {}).get(year)
            if not info:
                continue

            print(f"  {conf} {year}...")
            try:
                papers = (
                    fetch_api2(v2, info["id"])
                    if info["api"] == 2
                    else fetch_api1(v1, info["id"])
                )
            except Exception as e:
                print(f"    Error: {e}")
                continue

            matched = 0
            for p in papers:
                c = p.content
                texts = [get_field(c, f) for f in search_in]
                if any(matches(t, pattern) for t in texts):
                    results.append(
                        {
                            "conference": conf,
                            "year": str(year),
                            "title": get_field(c, "title"),
                            "abstract": get_field(c, "abstract"),
                            "authors": get_field(c, "authors"),
                            "keywords": get_field(c, "keywords"),
                            "pdf": f"https://openreview.net/pdf?id={p.id}",
                            "url": f"https://openreview.net/forum?id={p.id}",
                        }
                    )
                    matched += 1
            print(f"    {matched} matched\n")
            time.sleep(0.5)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# ACL ANTHOLOGY SEARCH
# ──────────────────────────────────────────────────────────────────────────────

def search_acl_anthology(venues, years, keywords, search_in, max_results=None):
    """Search ACL Anthology (ACL, EMNLP, NAACL, EACL, COLING)."""
    try:
        from acl_anthology import Anthology
    except ImportError:
        print("  [ACL Anthology] Missing: pip install acl-anthology")
        return []

    pattern = make_pattern(keywords)
    venue_set = {v.lower() for v in venues}
    year_set = {str(y) for y in years}
    results = []

    print("  [ACL] Loading ACL Anthology (cached after first run)...")
    try:
        anthology = Anthology.from_repo()
    except Exception as e:
        print(f"  [ACL] Error loading: {e}")
        return []

    print("  [ACL] Scanning papers...")
    matched = 0
    
    for paper in anthology.papers():
        if max_results is not None and matched >= max_results:
            break

        full_id = paper.full_id  # e.g. "2024.acl-long.42"
        parts = full_id.split(".")
        if len(parts) < 2:
            continue

        paper_year = parts[0]
        venue_slug = parts[1].split("-")[0].lower()

        if paper_year not in year_set:
            continue
        if venue_slug not in venue_set:
            continue

        title = str(paper.title) if paper.title else ""
        abstract = str(paper.abstract) if paper.abstract else ""

        texts = [title if f == "title" else abstract for f in search_in]
        if not any(matches(t, pattern) for t in texts):
            continue

        authors = ", ".join(
            f"{a.name.first} {a.name.last}".strip() for a in (paper.authors or [])
        )
        url = f"https://aclanthology.org/{full_id}"
        results.append(
            {
                "conference": venue_slug.upper(),
                "year": paper_year,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "keywords": "",
                "pdf": url + ".pdf",
                "url": url,
            }
        )
        matched += 1

    print(f"  [ACL] {matched} papers matched\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def search_conference_papers(
    keywords: list[str],
    years: list[int],
    conferences: list[str],
    search_in: list[str] | None = None,
    max_results: int = 10,
) -> dict:
    """
    Search conference papers across OpenReview and ACL Anthology.
    
    Args:
        keywords: List of search keywords (OR logic)
        years: List of publication years to include
        conferences: List of conferences (NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, EACL)
        search_in: Fields to search (default: ["title", "abstract"])
        max_results: Max results per conference
    
    Returns:
        dict with 'papers' list and metadata
    """
    if search_in is None:
        search_in = ["title", "abstract"]

    or_confs = [c for c in conferences if c in OPENREVIEW_CONFERENCES]
    acl_confs = [c for c in conferences if c.upper() in ACL_CONFERENCES]

    all_results = []
    truncated = False

    if or_confs and len(all_results) < max_results:
        remaining = max_results - len(all_results)
        openreview_results = search_openreview(or_confs, years, keywords, search_in)
        if len(openreview_results) > remaining:
            truncated = True
        all_results.extend(openreview_results[:remaining])

    if acl_confs and len(all_results) < max_results:
        remaining = max_results - len(all_results)
        acl_venues = [c.lower() for c in acl_confs]
        acl_results = search_acl_anthology(acl_venues, years, keywords, search_in, max_results=remaining)
        if len(acl_results) >= remaining:
            truncated = True
        all_results.extend(acl_results)
    elif acl_confs and len(all_results) >= max_results:
        truncated = True
    return {
        "total_found": len(all_results) + (100 if truncated else 0),  # Rough estimate if truncated
        "returned": len(all_results),
        "truncated": truncated,
        "papers": all_results,
    }


def handle_conference_paper_tool_call(tool_arguments: dict) -> str:
    """
    Handle a tool call for conference paper search.
    
    Args:
        tool_arguments: Parsed JSON arguments from LLM tool call
    
    Returns:
        JSON string with results
    """
    try:
        results = search_conference_papers(**tool_arguments)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "papers": [],
        })
