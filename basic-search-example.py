"""
auto_sci_api_with_links.py

Given a text query, choose suitable open scholarly/scientific APIs,
query them, and return aggregated results that include URLs.
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any


# -----------------------------
# API query functions
# -----------------------------

def query_openalex(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search works in OpenAlex.
    Each work has an 'id' like 'https://openalex.org/Wxxxx', and usually a primary location/source URL. [web:67][web:68]
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit,
        "mailto": "you@example.com",
    }
    r = requests.get(base_url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("results", []):
        work_id = item.get("id")  # canonical OpenAlex URL [web:67]
        # best available “read this” URL if present [web:68][web:71]
        url_for_reading = None
        if item.get("primary_location") and item["primary_location"].get("source"):
            url_for_reading = item["primary_location"]["source"].get("host_organization_url") \
                              or item["primary_location"].get("landing_page_url")
        elif item.get("locations"):
            # fallback to any landing_page_url in locations
            for loc in item["locations"]:
                if loc.get("landing_page_url"):
                    url_for_reading = loc["landing_page_url"]
                    break

        results.append({
            "source": "openalex",
            "title": item.get("title"),
            "year": item.get("publication_year"),
            "doi": item.get("doi"),
            "authors": [a.get("author", {}).get("display_name")
                        for a in item.get("authorships", [])],
            "openalex_url": work_id,
            "best_url": url_for_reading or work_id,
        })
    return results


def query_crossref(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search Crossref works and provide DOI + a resolvable URL. [web:61][web:72]
    """
    base_url = "https://api.crossref.org/works"
    params = {
        "query": query,
        "rows": limit,
        "mailto": "you@example.com",
    }
    r = requests.get(base_url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    items = data.get("message", {}).get("items", [])
    results = []
    for it in items:
        doi = it.get("DOI")
        # Official resolver URL pattern [web:61]
        doi_url = f"https://doi.org/{doi}" if doi else None

        # Crossref can also return a 'resource' URL for the landing page when selected.[web:72]
        resource_url = None
        if isinstance(it.get("resource"), dict):
            resource_url = it["resource"].get("primary")

        results.append({
            "source": "crossref",
            "title": (it.get("title") or [""])[0],
            "doi": doi,
            "year": it.get("issued", {}).get("date-parts", [[None]])[0][0],
            "authors": [
                f"{a.get('given','')} {a.get('family','')}".strip()
                for a in it.get("author", []) if isinstance(a, dict)
            ],
            "doi_url": doi_url,
            "best_url": resource_url or doi_url,
        })
    return results


def query_arxiv(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search arXiv and include both abstract and PDF URLs.
    The <id> element is the abstract page URL, and links with title='pdf' are the PDF. [web:75]
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
    }
    r = requests.get(base_url, params=params, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text.encode("utf-8"))
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    results = []
    for e in entries:
        title = e.findtext("atom:title", default="", namespaces=ns).strip()
        published = e.findtext("atom:published", default="", namespaces=ns)
        abstract_url = e.findtext("atom:id", default="", namespaces=ns)  # arxiv.org/abs/... [web:75]
        pdf_url = None
        for link in e.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href")
        authors = [a.findtext("atom:name", default="", namespaces=ns)
                   for a in e.findall("atom:author", ns)]
        results.append({
            "source": "arxiv",
            "title": title,
            "published": published,
            "abstract_url": abstract_url,
            "pdf_url": pdf_url,
            "best_url": pdf_url or abstract_url,
            "authors": authors,
        })
    return results


# -----------------------------
# Router
# -----------------------------

def choose_apis(query: str) -> list:
    q = query.lower()
    cs_math_physics = [
        "quantum", "relativity", "neural network", "machine learning",
        "deep learning", "graph theory", "astrophysics", "computer vision"
    ]
    bio_keywords = ["cancer", "gene", "genome", "protein", "clinical trial"]

    if any(k in q for k in cs_math_physics):
        return ["openalex", "arxiv", "crossref"]
    if any(k in q for k in bio_keywords):
        return ["openalex", "crossref"]
    return ["openalex", "crossref"]


def search_scientific_apis(query: str,
                           limit_per_api: int = 5) -> Dict[str, List[Dict[str, Any]]]:
    apis = choose_apis(query)
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    if "openalex" in apis:
        try:
            all_results["openalex"] = query_openalex(query, limit_per_api)
        except Exception as exc:
            all_results["openalex_error"] = [{"error": str(exc)}]

    if "crossref" in apis:
        try:
            all_results["crossref"] = query_crossref(query, limit_per_api)
        except Exception as exc:
            all_results["crossref_error"] = [{"error": str(exc)}]

    if "arxiv" in apis:
        try:
            all_results["arxiv"] = query_arxiv(query, limit_per_api)
        except Exception as exc:
            all_results["arxiv_error"] = [{"error": str(exc)}]

    return all_results


# -----------------------------
# CLI / example usage
# -----------------------------

def main():
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter a research topic or query: ").strip()

    results = search_scientific_apis(query, limit_per_api=5)

    for api_name, records in results.items():
        print(f"\n=== {api_name.upper()} ===")
        for r in records:
            title = r.get("title") or "NO TITLE"
            url = r.get("best_url") or r.get("abstract_url") or r.get("doi_url")
            print(f" - {title}")
            if url:
                print(f"   -> {url}")

if __name__ == "__main__":
    main()

