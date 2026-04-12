# Conference Paper Search Integration

This guide explains how to use the integrated conference paper search tool in your multi-agent RAG system.

## Overview

The web search agent now supports searching for papers across major ML/NLP conferences:

### OpenReview Venues (requires free account)
- **NeurIPS** (2021–2024)
- **ICML** (2021–2024)  
- **ICLR** (2021–2024)

### ACL Anthology Venues (no auth required)
- **ACL** (Association for Computational Linguistics)
- **EMNLP** (Empirical Methods in Natural Language Processing)
- **NAACL** (North American Chapter of ACL)
- **EACL** (European Chapter of ACL)

## Installation

### 1. Install dependencies

```bash
pip install openreview-py acl-anthology
```

Or use the full requirements:

```bash
pip install -r requirements.txt
```

### 2. Set up OpenReview credentials (optional but recommended)

To search NeurIPS/ICML/ICLR, you need a free OpenReview account:

1. **Create account**: Visit https://openreview.net/register
2. **Set environment variables**:

```bash
# macOS/Linux
export OPENREVIEW_USERNAME="your@email.com"
export OPENREVIEW_PASSWORD="yourpassword"

# Windows (Command Prompt)
set OPENREVIEW_USERNAME=your@email.com
set OPENREVIEW_PASSWORD=yourpassword

# Windows (PowerShell)
$env:OPENREVIEW_USERNAME="your@email.com"
$env:OPENREVIEW_PASSWORD="yourpassword"
```

**Or** create a `.env` file in the project root:

```env
OPENREVIEW_USERNAME=your@email.com
OPENREVIEW_PASSWORD=yourpassword
```

### 3. Verify the integration

Run a test query:

```bash
python -c "from conference_paper_search import search_conference_papers; \
results = search_conference_papers( \
    keywords=['diffusion models'], \
    years=[2023, 2024], \
    conferences=['NeurIPS', 'ICML']); \
print(results['returned'], 'papers found')"
```

## How It Works

### When the web agent activates:

1. **Query analysis**: The agent checks if your query mentions conference keywords (e.g., "NeurIPS", "paper search", "ACL", etc.)

2. **Tool invocation**: If conferences are relevant, the LLM gets access to the `search_conference_papers` tool

3. **LLM decides**: The model determines which conferences, years, and keywords are most relevant to your query

4. **Search execution**: Conference papers are retrieved from OpenReview or ACL Anthology

5. **Integration**: Results are combined with OpenAlex, Crossref, Semantic Scholar, and arXiv results

6. **Indexing**: Papers are added to the VectorDB for downstream extraction and synthesis

### Example Queries

✅ **Queries that trigger conference search:**
- "Find NeurIPS papers on transformer architectures"
- "What are recent ICML publications about reinforcement learning?"
- "Search ACL conference for papers on language models"
- "EMNLP 2024 papers on machine translation"

❌ **Queries that skip conference search (not relevant):**
- "What is machine learning?" (too general)
- "News about AI companies" (not academic)
- "How to install Python?" (not a literature search)

## Usage in Your RAG System

### Basic query through Streamlit UI:

```
User: "Show me NeurIPS 2024 papers on diffusion models"
↓
Router decides to activate web agent
↓
Web agent detects "NeurIPS" + "2024" + "diffusion"
↓
LLM calls: search_conference_papers(
  keywords=["diffusion models"],
  years=[2024],
  conferences=["NeurIPS"]
)
↓
Results integrated with other sources + indexed
↓
Reading extraction agent processes findings
↓
Summarizer generates final answer with citations
```

### Programmatic usage:

```python
from conference_paper_search import search_conference_papers

# Search multi-conference, multi-year
results = search_conference_papers(
    keywords=["prompt engineering", "large language models"],
    years=[2023, 2024],
    conferences=["ACL", "EMNLP", "ICLR"],
    search_in=["title", "abstract"],
    max_results=15
)

# Access results
print(f"Found {results['returned']} papers")
for paper in results['papers']:
    print(f"[{paper['conference']}] {paper['title']}")
    print(f"  Authors: {paper['authors']}")
    print(f"  URL: {paper['url']}\n")
```

## Limitation & Troubleshooting

### Common Issues

#### ❌ "Missing: pip install acl-anthology"

**Solution:** Install missing package:
```bash
pip install acl-anthology
```

#### ❌ OpenReview login fails

**Possible causes:**
- Credentials not set (check env vars: `echo $OPENREVIEW_USERNAME`)
- Wrong email/password
- Account not verified at openreview.net
- Network connectivity issue

**Solution:**
```bash
# Verify credentials are loaded
python -c "import os; print('User:', os.getenv('OPENREVIEW_USERNAME'))"

# Test login manually
python -c "import openreview; \
client = openreview.Client(baseurl='https://api.openreview.net', \
username='your@email.com', password='yourpassword')"
```

#### ❌ No conference papers returned

**Possible causes:**
- Query doesn't match indexed papers
- Wrong year specified (conferences only have historical data)
- LLM didn't recognize the tool should be called

**Solution:**
- Broaden keywords: "diffusion" instead of "probabilistic diffusion model"
- Check available years in `conference_paper_search.py` (currently 2021–2024)
- Explicit query: "Find NeurIPS 2024 papers on..."

#### ❌ ACL Anthology takes long on first run

**Expected behavior:** ACL Anthology (50MB) downloads and caches on first run (~30–60 sec). Subsequent runs are instant.

### Performance Tips

1. **Limit results**: Set `max_results` to 10–20 for faster responses
2. **Specify years**: Narrowing the year range makes searches faster
3. **Exact keywords**: Be specific ("transformer attention" vs "attention")
4. **Credential setup**: Set OpenReview creds to avoid timeout skipping

## Architecture

```
User Query
    ↓
Web Agent (streamlit_app.py)
    ├─→ Detect conference keywords
    ├─→ ChatOpenAI with tool option
    │   └─→ search_conference_papers tool (conference_paper_search.py)
    │       ├─→ OpenReview API (NeurIPS/ICML/ICLR)
    │       └─→ ACL Anthology (ACL/EMNLP/NAACL/EACL)
    ├─→ OpenAlex API
    ├─→ Crossref API
    ├─→ Semantic Scholar API
    └─→ arXiv API
    ↓
VectorDB (index papers for extraction)
    ↓
Reading Extraction Agent
    ↓
Orchestrator Agent (merge + weight evidence)
    ↓
Summarizer Agent
    ↓
Final Answer with Citations
```

## Supported Venues & Years

### OpenReview (via API)

| Venue | Years | Notes |
|-------|-------|-------|
| NeurIPS | 2021, 2022, 2023, 2024 | Neural Information Processing Systems |
| ICML | 2021, 2022, 2023, 2024 | International Conference on Machine Learning |
| ICLR | 2021, 2022, 2023, 2024 | International Conference on Learning Representations |

### ACL Anthology (via Web Scraping)

| Venue | Coverage | Notes |
|-------|----------|-------|
| ACL | 2021– | Main conference (all years available) |
| EMNLP | 2021– | Empirical methods track |
| NAACL | 2021– | North American regional |
| EACL | 2021– | European regional |
| COLING | 2021– | Computational linguistics |

## API Quotas & Rate Limits

- **OpenReview**: No strict quotas; brief 0.5-sec delays between requests
- **ACL Anthology**: Unlimited; uses local cache after first download
- **OpenAlex**: 10k requests/day per email
- **Crossref**: No API key needed; reasonable rate limits
- **Semantic Scholar**: API key recommended; 100 req/5min without key
- **arXiv**: No auth; ~3 sec per query recommended

## Advanced Configuration

### Custom score function for ranking

Edit `conference_paper_search.py` to add custom ranking:

```python
def rank_papers(papers, query):
    """Optional: rank papers by relevance score"""
    # Add scoring logic here
    return sorted(papers, key=lambda p: p.get("score", 0), reverse=True)
```

### Filtering results programmatically

```python
from conference_paper_search import search_conference_papers

results = search_conference_papers(
    keywords=["vision transformer"],
    years=[2024],
    conferences=["ICLR", "NeurIPS"],
)

# Filter by author, year, etc.
filtered = [p for p in results['papers'] 
            if "Dosovitskiy" in p['authors']]
```

## Contributing

To add support for more conferences:

1. `conference_paper_search.py` → add venue config in `OR_VENUES` dict
2. Implement corresponding `fetch_*` function
3. Update documentation with new conference details

## References

- [OpenReview API Docs](https://docs.openreview.net/)
- [ACL Anthology](https://aclanthology.org/)
- [acl-anthology Python Library](https://github.com/acl-org/acl-anthology)
- [openreview-py Library](https://github.com/openreview/openreview-py)

---

For questions or issues, check the agent logs in the Streamlit UI sidebar under "Activity Log".
