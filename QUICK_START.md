# Quick Start: Conference Paper Search

## 1-Minute Setup

```bash
# Install conference search dependencies
pip install openreview-py acl-anthology

# Optional: Set OpenReview credentials (for NeurIPS/ICML/ICLR search)
export OPENREVIEW_USERNAME="mchoudhu@g.syr.edu"
export OPENREVIEW_PASSWORD="Walantly!Aa12"

# Verify integration
python verify_conference_search.py

# Start the app
streamlit run streamlit_app.py
```

## Try It Now

In the Streamlit UI, paste one of these queries:

### Example 1: NeurIPS 2024
```
Find NeurIPS 2024 papers on diffusion models
```

### Example 2: Multiple conferences
```
Compare recent work on vision transformers from ICLR and ICML 2023-2024
```

### Example 3: ACL Anthology (no credentials needed)
```
Search ACL for papers on language model prompt engineering
```

### Example 4: EMNLP
```
What are the latest EMNLP papers on machine translation?
```

## How It Works

When you submit a query mentioning conferences:

1. **Query Detection**: System recognizes conference keywords (NeurIPS, ICML, ACL, etc.)
2. **LLM Decision**: AI decides if conference search is relevant
3. **Tool Call**: If yes, LLM calls `search_conference_papers` tool
4. **Results**: Papers fetched from OpenReview or ACL Anthology
5. **Integration**: All sources combined (OpenAlex, Crossref, Semantic Scholar, arXiv, Conference Papers)
6. **Processing**: Reading extraction → Orchestration → Final synthesis

## What's Supported

| Conference | Backend | Auth Required | Years Available |
|-----------|---------|---------------|-----------------|
| NeurIPS | OpenReview | Yes | 2021–2024 |
| ICML | OpenReview | Yes | 2021–2024 |
| ICLR | OpenReview | Yes | 2021–2024 |
| ACL | ACL Anthology | No | All |
| EMNLP | ACL Anthology | No | All |
| NAACL | ACL Anthology | No | All |
| EACL | ACL Anthology | No | All |

## Without OpenReview Credentials

The system still works! You'll get results from:
- ✓ ACL Anthology (ACL, EMNLP, NAACL, EACL, COLING)
- ✓ OpenAlex
- ✓ Crossref
- ✓ Semantic Scholar
- ✓ arXiv

But won't access: ✗ NeurIPS, ICML, ICLR (requires free account)

## Get OpenReview Credentials

1. Go to https://openreview.net/register
2. Create free account (confirmation email)
3. Set environment variables:
   ```bash
   export OPENREVIEW_USERNAME="your@email.com"
   export OPENREVIEW_PASSWORD="yourpassword"
   ```

Takes ~2 minutes total.

## Troubleshooting

### "Missing: pip install acl-anthology"
```bash
pip install -r requirements.txt
```

### "OpenReview login failed"
- Check credentials: `echo $OPENREVIEW_USERNAME`
- Verify at https://openreview.net (account exists + verified?)
- Try manual login: `python -c "import openreview; openreview.Client(baseurl='https://api.openreview.net', username='your@email.com', password='...'" 2>&1`

### "No papers found"
- Try simpler keywords: "transformer" not "deep hierarchical attention transformer"
- Expand years: `[2023, 2024]`
- Query explicitly: "Find papers on attention mechanisms from NeurIPS and ICML"

### "Takes too long (ACL)"
- Normal on first run: 50MB downloads to cache (~30–60 sec)
- Subsequent queries: instant (cached)

## Key Files

| File | Purpose |
|------|---------|
| `conference_paper_search.py` | Core search logic |
| `CONFERENCE_SEARCH_SETUP.md` | Full documentation |
| `INTEGRATION_SUMMARY.md` | Architecture overview |
| `verify_conference_search.py` | Diagnostics script |

## Under the Hood

```python
# Example of what the LLM sees:
from conference_paper_search import SEARCH_PAPERS_TOOL

# Tool definition available to LLM:
SEARCH_PAPERS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_conference_papers",
        "description": "Search academic papers across ML/NLP conferences...",
        "parameters": {
            "keywords": ["diffusion models"],
            "years": [2024],
            "conferences": ["NeurIPS", "ICML"],
            "search_in": ["title", "abstract"],
            "max_results": 10
        }
    }
}

# LLM can call this, receives results
```

## Advanced Usage

### Custom search programmatically

```python
from conference_paper_search import search_conference_papers

results = search_conference_papers(
    keywords=["reinforcement learning", "policy gradient"],
    years=[2023, 2024],
    conferences=["NeurIPS", "ICML", "ICLR"],
    max_results=20
)

for paper in results['papers']:
    print(f"{paper['conference']} {paper['year']}: {paper['title']}")
```

### Integrate into your own agents

```python
from langchain_openai import ChatOpenAI
from conference_paper_search import SEARCH_PAPERS_TOOL

model = ChatOpenAI(model="gpt-4o-mini")
response = model.invoke(
    messages,
    tools=[SEARCH_PAPERS_TOOL],
    tool_choice="auto"
)

# Handle tool calls
if response.tool_calls:
    for tc in response.tool_calls:
        # Tool call handling here
```

## Performance

- **OpenReview API**: 5–15 sec per request (real-time)
- **ACL Anthology**: instant (cached after first run)
- **Combined search**: 15–30 sec total for web agent
- **Full pipeline**: 30–60 sec (including extraction + synthesis)

## Supported Queries

✅ Works:
- "Find NeurIPS papers on diffusion"
- "ICML 2024 work on transformers"
- "Recent ACL papers on language models"
- "Search EMNLP for machine translation"
- "What's new in EACL about NLP?"

❌ Doesn't trigger (too general):
- "What is deep learning?" 
- "Tell me about AI"
- "Python tutorials"

---

**Questions?** See [CONFERENCE_SEARCH_SETUP.md](CONFERENCE_SEARCH_SETUP.md) for detailed docs.

**Test first?** Run `python verify_conference_search.py` to check integration.

**Issues?** Check Streamlit Activity Log for error details.
