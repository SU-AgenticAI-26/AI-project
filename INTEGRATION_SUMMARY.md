# Conference Paper Search Integration — Complete

## What Was Integrated

Your multi-agent RAG system now includes **conference paper search capability** that automatically activates when users search for papers from major ML/NLP conferences.

## Files Created/Modified

### New Files

| File | Purpose |
|------|---------|
| [`conference_paper_search.py`](conference_paper_search.py) | Core module with OpenReview + ACL Anthology search logic |
| [`CONFERENCE_SEARCH_SETUP.md`](CONFERENCE_SEARCH_SETUP.md) | Comprehensive setup & usage documentation |
| [`verify_conference_search.py`](verify_conference_search.py) | Verification script to test the integration |
| [`INTEGRATION_SUMMARY.md`](INTEGRATION_SUMMARY.md) | This file |

### Modified Files

| File | Changes |
|------|---------|
| [`streamlit_app.py`](streamlit_app.py) | • Added imports for conference_paper_search<br>• Enhanced web_agent with tool calling<br>• Detects conference keywords in queries<br>• Integrates conference results with existing APIs |
| [`requirements.txt`](requirements.txt) | • Added `openreview-py>=1.0.0`<br>• Added `acl-anthology>=0.9.0` |

## How It Works

### User Query Flow

```
User: "Find NeurIPS 2024 papers on diffusion models"
    ↓
Router Agent → Activates "web" agent
    ↓
Web Agent:
    1. Detects "NeurIPS" + "diffusion" keywords
    2. Offers tool to LLM: search_conference_papers()
    3. LLM decides to call tool with:
       • keywords: ["diffusion models"]
       • years: [2024]
       • conferences: ["NeurIPS"]
    4. Tool executes → returns 8 papers
    5. Combines with OpenAlex, Crossref, Semantic Scholar, arXiv
    6. All results indexed into VectorDB
    ↓
Reading Extraction Agent → Processes papers, extracts structured data
    ↓
Orchestrator → Merges + deduplicates + weights evidence
    ↓
Summarizer → Generates final answer with citations
```

### Key Features

✅ **Automatic Detection**: Recognizes conference names in queries (case-insensitive)

✅ **Tool-Based LLM Integration**: LLM can decide whether to call the search tool

✅ **Multi-Conference Search**: Search 1 or many conferences simultaneously

✅ **Multi-Year Support**: Search across multiple years (2021–2024)

✅ **VectorDB Integration**: Results fed into downstream extraction/synthesis

✅ **Fallback Support**: Works without OpenReview credentials (ACL Anthology searches will work)

✅ **Graceful Error Handling**: Missing dependencies don't crash the system

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Recommended) Set OpenReview credentials

```bash
export OPENREVIEW_USERNAME="your@email.com"
export OPENREVIEW_PASSWORD="yourpassword"
```

### 3. Verify integration

```bash
python verify_conference_search.py
```

Expected output: ✓ checks for modules, credentials, dependencies, and tool structure

### 4. Run the app

```bash
streamlit run streamlit_app.py
```

### 5. Try a conference search query

In the Streamlit UI, submit:
- "Find recent NeurIPS papers on vision transformers"
- "Show me ICLR 2024 work on neural architecture search"
- "Search ACL for papers on prompt engineering"
- "What's new in EMNLP 2024 for machine translation?"

## Supported Conferences & Years

### OpenReview (via API) — Requires Free Account
- **NeurIPS**: 2021, 2022, 2023, 2024
- **ICML**: 2021, 2022, 2023, 2024
- **ICLR**: 2021, 2022, 2023, 2024

### ACL Anthology (No Auth) — Always Available
- **ACL**: All years
- **EMNLP**: All years
- **NAACL**: All years
- **EACL**: All years

## Configuration

### Environment Variables

```bash
# OpenReview authentication
export OPENREVIEW_USERNAME="your_email@example.com"
export OPENREVIEW_PASSWORD="your_password"

# Optional: Semantic Scholar API key (for better results)
export SEMANTIC_SCHOLAR_API_KEY="your_api_key"
```

### Or use `.env` file

Create `.env` in project root:
```
OPENREVIEW_USERNAME=your@email.com
OPENREVIEW_PASSWORD=yourpassword
```

## Programmatic Usage

### Direct function call

```python
from conference_paper_search import search_conference_papers

results = search_conference_papers(
    keywords=["reinforcement learning", "policy gradient"],
    years=[2023, 2024],
    conferences=["NeurIPS", "ICML", "ICLR"],
    search_in=["title", "abstract"],
    max_results=15
)

# Iterate results
for paper in results['papers']:
    print(f"[{paper['conference']}] {paper['title']}")
    print(f"  Authors: {paper['authors']}")
    print(f"  URL: {paper['url']}\n")
```

### Integration with LLM agents

```python
from langchain_openai import ChatOpenAI
from conference_paper_search import SEARCH_PAPERS_TOOL, handle_conference_paper_tool_call

model = ChatOpenAI(model="gpt-4o-mini")

# Call with tool availability
response = model.invoke(
    messages,
    tools=[SEARCH_PAPERS_TOOL],
    tool_choice="auto"
)

# Handle tool calls
if response.tool_calls:
    for tc in response.tool_calls:
        result = handle_conference_paper_tool_call(json.loads(tc.function.arguments))
```

## Behavior Notes

### When tool is NOT called
- Query doesn't mention specific conferences
- Example: "How does attention work?" (too general)
- System falls back to OpenAlex + Crossref + Semantic Scholar + arXiv

### When tool IS called
- Query explicitly mentions: NeurIPS, ICML, ICLR, ACL, EMNLP, NAACL, EACL
- Or implied conference context: "Find recent ML conference papers on X"
- LLM determines conferences, years, and keywords to search

### Performance
- **ACL Anthology**: ~50MB cache downloaded on first run (~30–60 sec), then instant
- **OpenReview**: Real-time API calls (0.5 sec per request)
- Total agent response: 10–30 seconds depending on internet + LLM latency

## Troubleshooting

### "ModuleNotFoundError: No module named 'conference_paper_search'"

```bash
# Ensure you're in the project directory
cd /workspaces/AI-project

# Verify file exists
ls -la conference_paper_search.py

# Test import
python -c "import conference_paper_search"
```

### OpenReview authentication fails

```bash
# Check credentials are set
echo $OPENREVIEW_USERNAME
echo $OPENREVIEW_PASSWORD

# Test login manually
python -c "
import openreview
client = openreview.Client(
    baseurl='https://api.openreview.net',
    username='your@email.com',
    password='yourpassword'
)"
```

### No results returned

- Try simpler keywords: "transformer" instead of "hierarchical attention-based transformer"
- Expand years: `[2023, 2024]` instead of just `[2024]`
- Specify all relevant conferences: `["NeurIPS", "ICML", "ICLR"]`
- Check available papers: https://openreview.net or https://aclanthology.org

### ACL Anthology takes too long

Expected on first run (50MB download + parsing). Subsequent calls use cache. For immediate results, disable ACL search initially in `web_agent()`.

## Architecture Changes

### Before Integration
```
Query → Router → [WebAgent (4 APIs)] → Extraction → Synthesis
```

### After Integration
```
Query → Router → [WebAgent (4 APIs + Tool) + LLM Decision] → Extraction → Synthesis
                      ↓
              Tool: search_conference_papers()
                  ├→ OpenReview (NeurIPS/ICML/ICLR)
                  └→ ACL Anthology (ACL/EMNLP/NAACL/EACL)
```

## API Rate Limits

- **OpenReview**: No strict limit; 0.5 sec delay recommended between requests
- **ACL Anthology**: Unlimited (local cache)
- **OpenAlex**: 10k/day per email
- **Crossref**: Reasonable rate limit (no key needed)
- **Semantic Scholar**: 100 req/5min without key
- **arXiv**: ~3 sec per query recommended

## Next Steps

1. **Test the integration**: Run `verify_conference_search.py`
2. **Set credentials**: Add OpenReview login (optional but recommended)
3. **Try queries**: Use Streamlit UI to test with conference-related questions
4. **Monitor activity**: Check "Activity Log" in sidebar for execution details
5. **Customize**: Modify `conference_paper_search.py` to add more venues or custom ranking

## Customization

### Add a new OpenReview venue

Edit `conference_paper_search.py`:

```python
OR_VENUES = {
    ...
    "ICAI": {
        2024: {"id": "ICAI.cc/2024/Conference", "api": 2},
        2023: {"id": "ICAI.cc/2023/Conference", "api": 2},
    },
}
```

Then add to `OPENREVIEW_CONFERENCES`:
```python
OPENREVIEW_CONFERENCES = {"NeurIPS", "ICML", "ICLR", "ICAI"}
```

### Filter by author or institution

```python
results = search_conference_papers(...)
filtered = [p for p in results['papers'] 
            if "Bengio" in p['authors']]
```

## References

- **OpenReview API**: https://docs.openreview.net/
- **ACL Anthology**: https://aclanthology.org/
- **acl-anthology Library**: https://github.com/acl-org/acl-anthology
- **openreview-py Library**: https://github.com/openreview/openreview-py

---

## Support

For detailed setup, usage examples, and troubleshooting:
→ See [`CONFERENCE_SEARCH_SETUP.md`](CONFERENCE_SEARCH_SETUP.md)

For verification and diagnostics:
→ Run [`verify_conference_search.py`](verify_conference_search.py)

Questions or feedback:
→ Check the Streamlit app's Activity Log for debug information
→ Review error messages in the terminal where you ran `streamlit run`

---

**Integration Status**: ✅ **Complete**

Your RAG system now seamlessly combines web APIs with targeted conference repository searches for comprehensive paper discovery!
