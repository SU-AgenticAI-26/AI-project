"""
tools/sources.py — Academic source registry and search backends.

Sources
───────
  semantic_scholar — broad STEM/CS, citation graph, good abstracts
  arxiv            — CS, physics, math, quantitative bio; preprints
  pubmed           — biomedicine, clinical, pharmacology, genomics
  openalex         — cross-disciplinary open-access; 250M+ works
  europe_pmc       — life sciences, medicine; complements PubMed
  crossref         — broadest DOI coverage incl. humanities and law
                     NOTE: abstracts rarely available
  dblp             — CS bibliography (ACM/IEEE/USENIX)
                     NOTE: metadata only, no abstracts

Standard paper dict format:
  {source, paper_id, title, authors, abstract, year,
   citation_count, pdf_url, url, has_abstract}

has_abstract=False: paper is kept in the corpus but the summarisation
step will not call the LLM (there is nothing to summarise).
"""

import re
import time
import xml.etree.ElementTree as ET
from typing import Optional
import requests

# ── Registry ──────────────────────────────────────────────────────────────────

SOURCE_REGISTRY = {
    "semantic_scholar": {
        "name":          "Semantic Scholar",
        "description":   "Broad STEM/CS index with citation counts and open-access PDFs.",
        "best_for":      ["computer science", "AI", "machine learning", "NLP",
                          "mathematics", "physics", "engineering", "biology"],
        "not_for":       ["humanities", "law", "social sciences"],
        "has_abstracts": True,
        "has_citations": True,
        "priority":      1,
    },
    "arxiv": {
        "name":          "arXiv",
        "description":   "Preprint server for CS, AI/ML, physics, math, statistics.",
        "best_for":      ["AI", "machine learning", "deep learning", "theoretical CS",
                          "physics", "mathematics", "statistics"],
        "not_for":       ["clinical medicine", "humanities", "social sciences"],
        "has_abstracts": True,
        "has_citations": False,
        "priority":      2,
    },
    "pubmed": {
        "name":          "PubMed (NCBI)",
        "description":   "Authoritative biomedical database: medicine, clinical, pharmacology, genomics, neuroscience.",
        "best_for":      ["medicine", "clinical research", "pharmacology", "genomics",
                          "neuroscience", "public health", "epidemiology", "biology",
                          "biochemistry", "mental health", "nursing", "dentistry"],
        "not_for":       ["pure CS/AI", "humanities", "law", "civil engineering"],
        "has_abstracts": True,
        "has_citations": False,
        "priority":      2,
    },
    "openalex": {
        "name":          "OpenAlex",
        "description":   "Cross-disciplinary open-access index: 250M+ works. Strong for social sciences, education, economics.",
        "best_for":      ["education", "social sciences", "economics", "psychology",
                          "interdisciplinary", "environmental science", "sociology",
                          "political science", "public policy"],
        "not_for":       ["very recent preprints", "deep CS specialisation"],
        "has_abstracts": True,
        "has_citations": True,
        "priority":      3,
    },
    "europe_pmc": {
        "name":          "Europe PMC",
        "description":   "European biomedical database: life sciences, medicine, biochemistry. Complements PubMed.",
        "best_for":      ["life sciences", "medicine", "biology", "biochemistry",
                          "clinical trials", "bioinformatics", "genetics"],
        "not_for":       ["CS/AI", "humanities", "social sciences", "engineering"],
        "has_abstracts": True,
        "has_citations": False,
        "priority":      3,
    },
    "crossref": {
        "name":          "CrossRef",
        "description":   "Broadest DOI coverage across all disciplines. Abstracts rarely available — use for metadata breadth.",
        "best_for":      ["humanities", "law", "political science", "sociology",
                          "history", "linguistics", "civil engineering",
                          "architecture", "niche or interdisciplinary fields"],
        "not_for":       ["queries needing full abstracts"],
        "has_abstracts": False,
        "has_citations": False,
        "priority":      4,
    },
    "dblp": {
        "name":          "DBLP",
        "description":   "CS bibliography: ACM, IEEE, USENIX. Metadata only — no abstracts.",
        "best_for":      ["computer science", "software engineering", "databases",
                          "networking", "security", "programming languages"],
        "not_for":       ["queries needing abstracts — metadata only"],
        "has_abstracts": False,
        "has_citations": False,
        "priority":      5,
    },
}


def list_available_sources() -> dict:
    return SOURCE_REGISTRY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


# ── Semantic Scholar ──────────────────────────────────────────────────────────

def _search_semantic_scholar(query, max_results, api_key=None):
    fields  = "title,authors,abstract,year,citationCount,openAccessPdf"
    headers = {"x-api-key": api_key} if api_key else {}
    params  = {"query": query, "limit": min(max_results, 15), "fields": fields}
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params, headers=headers, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            resp.raise_for_status()
            out = []
            for p in resp.json().get("data", []):
                pdf_url  = (p.get("openAccessPdf") or {}).get("url")
                abstract = _clean(p.get("abstract") or "")
                out.append({
                    "source": "semantic_scholar",
                    "paper_id": p.get("paperId", ""),
                    "title": _clean(p.get("title", "Untitled")),
                    "authors": [a.get("name", "") for a in p.get("authors", [])],
                    "abstract": abstract,
                    "year": p.get("year"),
                    "citation_count": p.get("citationCount", 0) or 0,
                    "pdf_url": pdf_url,
                    "url": f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                    "has_abstract": bool(abstract),
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── arXiv ─────────────────────────────────────────────────────────────────────

def _search_arxiv(query, max_results):
    NS     = "http://www.w3.org/2005/Atom"
    params = {"search_query": f"all:{query}", "max_results": min(max_results, 15),
              "sortBy": "relevance", "sortOrder": "descending"}
    for attempt in range(3):
        try:
            resp = requests.get("https://export.arxiv.org/api/query",
                                params=params, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            out  = []
            for entry in root.findall(f"{{{NS}}}entry"):
                def _t(tag):
                    el = entry.find(f"{{{NS}}}{tag}")
                    return _clean(el.text) if el is not None and el.text else ""
                url      = _t("id")
                year_s   = _t("published")
                authors  = [a.find(f"{{{NS}}}name").text
                             for a in entry.findall(f"{{{NS}}}author")
                             if a.find(f"{{{NS}}}name") is not None]
                pdf_url  = url.replace("/abs/", "/pdf/") + ".pdf" if "/abs/" in url else None
                abstract = _t("summary")
                out.append({
                    "source": "arxiv", "paper_id": url, "title": _t("title"),
                    "authors": authors, "abstract": abstract,
                    "year": int(year_s[:4]) if year_s else None,
                    "citation_count": 0, "pdf_url": pdf_url, "url": url,
                    "has_abstract": bool(abstract),
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── PubMed ────────────────────────────────────────────────────────────────────

def _search_pubmed(query, max_results):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    for attempt in range(3):
        try:
            sr = requests.get(f"{base}/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": min(max_results, 15),
                        "retmode": "json", "sort": "relevance"}, timeout=15)
            if sr.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            sr.raise_for_status()
            ids = sr.json().get("esearchresult", {}).get("idlist", [])
            if not ids: return []
            time.sleep(0.35)
            fr = requests.get(f"{base}/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(ids),
                        "retmode": "xml", "rettype": "abstract"}, timeout=20)
            fr.raise_for_status()
            root = ET.fromstring(fr.text)
            out  = []
            for art in root.findall(".//PubmedArticle"):
                pmid = (art.find(".//PMID") or type("", (), {"text": ""})()).text or ""
                tel  = art.find(".//ArticleTitle")
                title = _clean(tel.text or "") if tel is not None else "Untitled"
                abstract = " ".join(
                    _clean(el.text or "")
                    for el in art.findall(".//AbstractText") if el.text)
                authors = []
                for a in art.findall(".//Author"):
                    last = a.findtext("LastName", ""); first = a.findtext("ForeName", "")
                    if last: authors.append(f"{first} {last}".strip())
                yel  = art.find(".//PubDate/Year")
                year = int(yel.text) if yel is not None and yel.text else None
                out.append({
                    "source": "pubmed", "paper_id": f"pmid:{pmid}", "title": title,
                    "authors": authors, "abstract": abstract, "year": year,
                    "citation_count": 0, "pdf_url": None,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "has_abstract": bool(abstract),
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── OpenAlex ──────────────────────────────────────────────────────────────────

def _reconstruct_abstract(inv):
    if not inv: return ""
    pos = {}
    for word, positions in inv.items():
        for p in positions:
            pos[p] = word
    return " ".join(pos[i] for i in sorted(pos))


def _search_openalex(query, max_results):
    params = {
        "search": query, "per-page": min(max_results, 15),
        "sort": "relevance_score:desc", "filter": "has_abstract:true",
        "select": "id,title,authorships,abstract_inverted_index,publication_year,"
                  "cited_by_count,open_access,doi",
        "mailto": "research-assistant@example.com",
    }
    for attempt in range(3):
        try:
            resp = requests.get("https://api.openalex.org/works",
                                params=params, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            resp.raise_for_status()
            out = []
            for w in resp.json().get("results", []):
                authors  = [a.get("author", {}).get("display_name", "")
                            for a in w.get("authorships", [])]
                oa       = w.get("open_access", {})
                doi      = w.get("doi") or ""
                url      = (f"https://doi.org/{doi.replace('https://doi.org/','')}"
                            if doi else w.get("id", ""))
                abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
                out.append({
                    "source": "openalex", "paper_id": w.get("id", ""),
                    "title": _clean(w.get("title") or "Untitled"),
                    "authors": authors, "abstract": abstract,
                    "year": w.get("publication_year"),
                    "citation_count": w.get("cited_by_count", 0) or 0,
                    "pdf_url": oa.get("oa_url"), "url": url,
                    "has_abstract": bool(abstract),
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── Europe PMC ────────────────────────────────────────────────────────────────

def _search_europe_pmc(query, max_results):
    params = {"query": query, "pageSize": min(max_results, 15),
              "format": "json", "resultType": "core", "sort": "RELEVANCE"}
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params=params, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            resp.raise_for_status()
            out = []
            for r in resp.json().get("resultList", {}).get("result", []):
                pmid     = r.get("pmid", r.get("id", ""))
                authors  = [a.strip() for a in r.get("authorString", "").split(",") if a.strip()]
                abstract = _clean(r.get("abstractText", ""))
                out.append({
                    "source": "europe_pmc", "paper_id": f"epmc:{pmid}",
                    "title": _clean(r.get("title", "Untitled")),
                    "authors": authors, "abstract": abstract,
                    "year": r.get("pubYear"),
                    "citation_count": r.get("citedByCount", 0) or 0,
                    "pdf_url": (f"https://europepmc.org/articles/{r.get('pmcid', '')}/pdf"
                                if r.get("pmcid") else None),
                    "url": f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
                    "has_abstract": bool(abstract),
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── CrossRef ──────────────────────────────────────────────────────────────────

def _search_crossref(query, max_results):
    params = {"query": query, "rows": min(max_results, 15), "sort": "relevance",
              "select": "DOI,title,author,abstract,published,is-referenced-by-count,link",
              "mailto": "research-assistant@example.com"}
    for attempt in range(3):
        try:
            resp = requests.get("https://api.crossref.org/works",
                                params=params, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            resp.raise_for_status()
            out = []
            for w in resp.json().get("message", {}).get("items", []):
                titles  = w.get("title", [])
                title   = _clean(titles[0]) if titles else "Untitled"
                authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                           for a in w.get("author", [])]
                doi     = w.get("DOI", "")
                pub     = w.get("published", {}).get("date-parts", [[]])[0]
                year    = pub[0] if pub else None
                links   = w.get("link", [])
                pdf_url = next((l["URL"] for l in links
                                if "pdf" in l.get("content-type", "")), None)
                abstract = re.sub(r"<[^>]+>", " ", _clean(w.get("abstract", "")))
                out.append({
                    "source": "crossref", "paper_id": f"doi:{doi}", "title": title,
                    "authors": authors, "abstract": abstract, "year": year,
                    "citation_count": w.get("is-referenced-by-count", 0) or 0,
                    "pdf_url": pdf_url,
                    "url": f"https://doi.org/{doi}" if doi else "",
                    "has_abstract": bool(abstract),
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── DBLP ──────────────────────────────────────────────────────────────────────

def _search_dblp(query, max_results):
    params = {"q": query, "format": "json", "h": min(max_results, 15)}
    for attempt in range(3):
        try:
            resp = requests.get("https://dblp.org/search/publ/api",
                                params=params, timeout=15)
            if resp.status_code in (429, 503):
                time.sleep(2 ** attempt); continue
            resp.raise_for_status()
            hits = resp.json().get("result", {}).get("hits", {}).get("hit", [])
            out  = []
            for hit in hits:
                info = hit.get("info", {})
                ar   = info.get("authors", {}).get("author", [])
                if isinstance(ar, str): authors = [ar]
                elif isinstance(ar, dict): authors = [ar.get("text", "")]
                else: authors = [a.get("text","") if isinstance(a, dict) else str(a) for a in ar]
                yr = info.get("year")
                out.append({
                    "source": "dblp", "paper_id": info.get("key", ""),
                    "title": _clean(info.get("title", "Untitled")),
                    "authors": authors, "abstract": "",
                    "year": int(yr) if yr else None,
                    "citation_count": 0, "pdf_url": info.get("ee"),
                    "url": info.get("url", ""), "has_abstract": False,
                })
            return out
        except Exception:
            if attempt == 2: return []
            time.sleep(2 ** attempt)
    return []


# ── Dispatcher ────────────────────────────────────────────────────────────────

def search_source(source_id: str, query: str, max_results: int = 10,
                  api_key: Optional[str] = None) -> list:
    import os
    if source_id == "semantic_scholar":
        return _search_semantic_scholar(
            query, max_results, api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    elif source_id == "arxiv":       return _search_arxiv(query, max_results)
    elif source_id == "pubmed":      return _search_pubmed(query, max_results)
    elif source_id == "openalex":    return _search_openalex(query, max_results)
    elif source_id == "europe_pmc":  return _search_europe_pmc(query, max_results)
    elif source_id == "crossref":    return _search_crossref(query, max_results)
    elif source_id == "dblp":        return _search_dblp(query, max_results)
    else:                            return []


# ── Citation graph helpers ─────────────────────────────────────────────────────

def get_paper_references(paper_id: str, api_key: str = None,
                          max_refs: int = 10) -> list:
    """
    Fetch up to max_refs references for a Semantic Scholar paper_id.
    Returns list of {title, paper_id, year} dicts.
    Only works for papers retrieved from Semantic Scholar.
    """
    if not paper_id:
        return []
    import os
    headers = {"x-api-key": api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")}
    headers = {k: v for k, v in headers.items() if v}
    url     = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    params  = {"fields": "title,year", "limit": max_refs}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=12)
        if resp.status_code in (404, 400):
            return []
        resp.raise_for_status()
        out = []
        for item in resp.json().get("data", []):
            ref = item.get("citedPaper", {})
            title = _clean(ref.get("title", ""))
            if title:
                out.append({
                    "title":    title,
                    "paper_id": ref.get("paperId", ""),
                    "year":     ref.get("year"),
                })
        return out
    except Exception:
        return []


def build_citation_edges(paper_summaries: list, max_refs_per_paper: int = 8,
                          api_key: str = None) -> list:
    """
    For each SS paper in the corpus, fetch its references and record an edge
    from the corpus paper to any reference whose title also appears in the corpus
    (shared citation), plus edges to external references cited by ≥2 corpus papers.

    Returns list of {source, target, in_corpus} dicts where source and target
    are paper titles.
    """
    import time, os
    if not api_key:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

    corpus_titles = {p["title"].lower().strip() for p in paper_summaries}
    # paper_id → title for SS papers
    ss_papers = [p for p in paper_summaries
                 if p.get("source") == "semantic_scholar" and p.get("paper_id")]

    # ref_title → set of corpus paper titles that cite it
    ref_cited_by: dict = {}

    for paper in ss_papers:
        refs = get_paper_references(paper["paper_id"], api_key, max_refs_per_paper)
        time.sleep(0.4)  # be polite to SS API
        for ref in refs:
            rt = ref["title"]
            ref_cited_by.setdefault(rt, set()).add(paper["title"])

    edges = []
    seen  = set()

    for ref_title, citing_papers in ref_cited_by.items():
        # Edge: corpus paper → reference (if reference is also in corpus)
        ref_in_corpus = ref_title.lower().strip() in corpus_titles
        for cp in citing_papers:
            key = (cp, ref_title)
            if key not in seen:
                seen.add(key)
                edges.append({
                    "source":    cp,
                    "target":    ref_title,
                    "in_corpus": ref_in_corpus,
                })
        # If ≥2 corpus papers cite the same external reference, show those
        # corpus papers sharing that reference
        if len(citing_papers) >= 2 and not ref_in_corpus:
            titles = sorted(citing_papers)
            for i, a in enumerate(titles):
                for b in titles[i + 1:]:
                    key = (a, b, ref_title)
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "source":       a,
                            "target":       b,
                            "shared_via":   ref_title,
                            "in_corpus":    False,
                        })

    return edges
