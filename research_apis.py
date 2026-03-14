# Unified API module combining all research data sources

import requests
from defusedxml import ElementTree as ET
from typing import List, Dict, Any
import arxiv
import os

from secrets_manager import get_api_key

class ResearchAPIs:
    def __init__(self):
        self.nasa_key = get_api_key('nasa')
        self.semantic_key = get_api_key('semantic_scholar')

    def query_openalex(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search works in OpenAlex.
        Each work has an 'id' like 'https://openalex.org/Wxxxx', and usually a primary location/source URL.
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
            work_id = item.get("id")  # canonical OpenAlex URL
            # best available "read this" URL if present
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

    def query_crossref(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search Crossref works and provide DOI + a resolvable URL.
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
            # Official resolver URL pattern
            doi_url = f"https://doi.org/{doi}" if doi else None

            # Crossref can also return a 'resource' URL for the landing page when selected.
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

    def query_arxiv(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv and include both abstract and PDF URLs."""
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
            abstract_url = e.findtext("atom:id", default="", namespaces=ns)  # arxiv.org/abs/...
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

    def choose_apis(self, query: str) -> list:
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

    def search_scholarly(self, query: str, limit_per_api: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        apis = self.choose_apis(query)
        all_results: Dict[str, List[Dict[str, Any]]] = {}

        if "openalex" in apis:
            try:
                all_results["openalex"] = self.query_openalex(query, limit_per_api)
            except Exception as exc:
                all_results["openalex_error"] = [{"error": str(exc)}]

        if "crossref" in apis:
            try:
                all_results["crossref"] = self.query_crossref(query, limit_per_api)
            except Exception as exc:
                all_results["crossref_error"] = [{"error": str(exc)}]

        if "arxiv" in apis:
            try:
                all_results["arxiv"] = self.query_arxiv(query, limit_per_api)
            except Exception as exc:
                all_results["arxiv_error"] = [{"error": str(exc)}]

        return all_results

    def search_arxiv(self, query: str, max_results: int = 5):
        """Arxiv search using official library"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        return list(client.results(search))

    def search_semantic_scholar(self, query: str, year: str = "2020-"):
        """Semantic Scholar search"""
        if not self.semantic_key:
            return []
        url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
        params = {
            "query": query,
            "fields": "title,year,citationCount,authors,openAccessPdf",
            "year": year
        }
        headers = {"x-api-key": self.semantic_key}
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except requests.exceptions.RequestException:
            return []

    def get_nasa_apod(self):
        """NASA Astronomy Picture of the Day"""
        if not self.nasa_key:
            return None
        try:
            import nasapy
            nasa = nasapy.Nasa(self.nasa_key)
            return nasa.picture_of_the_day()
        except ImportError:
            return {"error": "nasapy not installed"}
        except Exception as e:
            return {"error": str(e)}