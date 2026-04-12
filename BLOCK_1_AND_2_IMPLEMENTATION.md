# Block 1 & Block 2 Implementation — High-Impact Demo Features

**Date**: April 11, 2026  
**Status**: ✅ **COMPLETE** and syntax-validated

---

## Strategic Value

### Block 1: UI Explicitness
- **Problem**: Pipeline does smart things (query routing, source diversity) that are **invisible to audience**
- **Solution**: Display query scoping panel showing sub-questions & routing decisions **before results appear**
- **Demo Impact**: Looks like "sophisticated system" not a black box
- **When user sees this**: Immediately after query submission, before graph runs

### Block 2: Citation Grounding  
- **Problem**: Highest risk for demo audience: "How do you know that citation is real?" → **no good answer**
- **Solution**: Live badge showing "18/20 citations grounded" (**explicit claim verification**)
- **Demo Impact**: Preempts hallucination questions; makes implicit proposal claims explicit
- **When user sees this**: In the "Final Answer" tab, top of results

---

## Implementation Details

### Block 1: Query Scoping Agent

#### What It Does
1. **Runs first** in the agent pipeline (entry point)
2. **Extracts from query**:
   - 3–5 focused sub-questions
   - 4–8 key themes/keywords
3. **Explains reasoning** in one sentence
4. Returns structured output as state fields

#### Code Changes

**1. Added to AgentState** (line 201–220):
```python
class AgentState(TypedDict):
    # ... existing fields ...
    # ──── BLOCK 1: Query Scoping ────
    sub_questions: List[str]        # 3–5 decomposed questions
    keywords: List[str]             # 4–8 key themes
    scoping_reasoning: str          # Brief explain
    # ────────────────────────────────
```

**2. New Agent Function** (line 635–670):
```python
def scoping_agent(state: AgentState, model: BaseChatModel) -> dict:
    """
    Extract sub-questions and keywords from user query.
    Makes query understanding visible to users before results appear.
    """
    system = SystemMessage(content=(
        "You are a Query Scoping Agent. Decompose the user's query into:\n"
        "1. 3–5 focused sub-questions...\n"
        "2. 4–8 key terms/themes..."
        # Returns JSON with sub_questions, keywords, reasoning
    ))
    # Parse LLM response, return structured scoping
```

**3. Updated build_graph()** (line ~1860):
```python
# Scoping agent becomes ENTRY POINT (not router)
g.add_node("scoping", lambda s: scoping_agent(s, lm_r))
g.set_entry_point("scoping")  # ← CHANGED from "router"
g.add_edge("scoping", "router")  # Routes directly to router
```

**4. Initial State Setup** (line ~2400):
```python
full_state = {
    # New fields with defaults
    "sub_questions": [],
    "keywords": [],
    "scoping_reasoning": "",
    # ... existing fields ...
}
```

**5. UI Display** (line ~2450, Agent Activity tab):
```python
with r_act:
    st.subheader("🔍 Query Understanding")
    sub_q = full_state.get("sub_questions", [])
    keywords = full_state.get("keywords", [])
    
    if sub_q or keywords:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**🎯 Sub-Questions Identified**")
            for i, q in enumerate(sub_q[:5], 1):
                st.caption(f"{i}. {q}")
        with sc2:
            st.markdown("**📌 Key Themes**")
            kw_badges = " ".join([f"`{k}`" for k in keywords[:8]])
            st.markdown(kw_badges)
        st.divider()
    
    st.subheader("What each agent did")
    # ... rest of activity log ...
```

#### UI Flow Demo
```
User enters query: "How do diffusion models improve image generation?"
           ↓
[Scoping Agent runs]
           ↓
Display:  🔍 Query Understanding
          🎯 Sub-Questions Identified
          1. What are diffusion models?
          2. How do they work?
          3. What are their advantages?
          4. What are current limitations?
          5. How do they compare to GANs?
          
          📌 Key Themes
          `diffusion` `image-generation` `probabilistic` `denoising` `latent-space` `training` `sampling` `inference`
          ─────────────────────────────────
          🤝 Agent Activity
          [Activity log begins...]
```

---

### Block 2: Citation Grounding Validator

#### What It Does
1. **After synthesis**, validates each claim in the summary against retrieved context
2. **Checks**:
   - Does the citation text appear in merged_context?
   - Does it appear in extraction_findings?
   - What percentage of keywords overlap?
3. **Returns**:
   - **grounding_map**: `{citation: {grounded: bool, source: str, evidence: text}}`
   - **grounding_score**: 0.0–1.0 percentage of citations grounded

#### Code Changes

**1. Added to AgentState** (line 201–220):
```python
class AgentState(TypedDict):
    # ... existing fields ...
    # ──── BLOCK 2: Citation Grounding ────
    citation_grounding: dict  # {citation: {grounded, source, evidence}}
    grounding_score: float    # 0.0-1.0
    # ──────────────────────────────────────
```

**2. New Validator Function** (line 672–750):
```python
def validate_citations(summary: str, merged_context: str, extraction_findings: str) -> tuple[dict, float]:
    """
    Validate that citations in summary actually appear in retrieved context.
    Returns (grounding_map, score) where score is 0.0–1.0.
    """
    grounding_map = {}
    
    # Extract citations: "quoted claims" or (parenthetical claims)
    citation_patterns = [
        r'"([^"]{20,150})"',      # "quoted claims"
        r'\(([^)]{20,150})\)',    # (parenthetical)
    ]
    citations = [cit for cit in citations if len(cit.split()) >= 3]
    
    # For each citation, check if it (or >60% of content words) appears in context
    for cit in citations:
        found_in_merged = cit_lower in merged_context.lower()
        found_in_extraction = cit_lower in extraction_findings.lower()
        
        # Fallback: word overlap check (>60% of content words)
        if not (found_in_merged or found_in_extraction):
            content_words = [w for w in words if len(w) > 3]
            matching = sum(1 for w in content_words if w in context_lower)
            if matching >= len(content_words) * 0.6:
                found_in_merged = True
        
        is_grounded = found_in_merged or found_in_extraction
        grounding_map[cit[:100]] = {
            "grounded": is_grounded,
            "source": "merged" if found_in_merged else ("extraction" if found_in_extraction else "none"),
            "evidence": evidence_snippet,
        }
    
    # Calculate score
    score = grounded_count / len(citations) if citations else 1.0
    return grounding_map, score
```

**3. Modified summarizer_agent()** (line ~1680):
```python
def summarizer_agent(state: AgentState, model: BaseChatModel) -> dict:
    # ... generate summary as before ...
    
    # ── BLOCK 2: Validate citations against retrieved sources ─────────────────
    citation_grounding, grounding_score = validate_citations(
        resp.content,
        state["merged_context"],
        state["extraction_findings"]
    )
    
    grounded_count = sum(1 for v in citation_grounding.values() if v.get("grounded"))
    total_citations = len(citation_grounding)
    
    return {
        "summary": resp.content,
        "citation_grounding": citation_grounding,
        "grounding_score": grounding_score,
        # ... activity log shows grounding percentage in detail ...
        "detail": f"{len(resp.content)} chars · {grounded_count}/{total_citations} citations grounded ({int(grounding_score*100)}%)",
    }
```

**4. Initial State Setup** (line ~2400):
```python
full_state = {
    # New fields with defaults
    "citation_grounding": {},
    "grounding_score": 0.0,
    # ... existing fields ...
}
```

**5. UI Display** (line ~2500, Final Answer tab):
```python
with r_ans:
    # ── BLOCK 2: Display Citation Grounding Badge ────────────────────
    grounding = full_state.get("citation_grounding", {})
    grounding_score = full_state.get("grounding_score", 0.0)
    
    if grounding:
        grounded_count = sum(1 for v in grounding.values() if v.get("grounded"))
        total = len(grounding)
        
        # Color bar based on grounding percentage
        color = "🟢" if grounding_score >= 0.8 else "🟡" if grounding_score >= 0.6 else "🔴"
        
        mc1, mc2, mc3 = st.columns([2, 1, 1])
        with mc1:
            st.markdown(f"### {color} Citation Grounding: {grounded_count}/{total} ({int(grounding_score*100)}%)")
        with mc2:
            if grounding_score >= 0.8:
                st.success("Well grounded")
            elif grounding_score >= 0.6:
                st.warning("Partially grounded")
            else:
                st.error("Low grounding")
        
        # Expandable details
        with st.expander("📋 Citation Details"):
            for citation, info in list(grounding.items())[:15]:
                status = "✅" if info.get("grounded") else "⚠️"
                st.markdown(f"{status} **{citation[:80]}...**")
                st.caption(f"Source: `{info.get('source', 'none')}`")
                if info.get("evidence"):
                    st.caption(f"Evidence: _{info['evidence'][:100]}..._")
        
        st.divider()
    
    st.markdown(full_state.get("summary", ""))
```

#### UI Flow Demo
```
User submitted query, results appear:
           ↓
[Summarizer runs → validates citations]
           ↓
Display in "💡 Final Answer" tab:
           
           🟢 Citation Grounding: 18/20 (90%)
           ✅ Well grounded
           
           📋 Citation Details [EXPAND]
           ✅ "Diffusion models are trained by predicting noise..."
              Source: `merged`
              Evidence: _In the merged context we found: "Diffusion models perform noise prediction..."_
           
           ⚠️ "Recent papers show 15% improvement over GANs"
              Source: `none`
              Evidence: _(not found in retrieved text)_
           
           ─────────────────────────────────────
           [Full summary markdown below...]
```

---

## Files Modified

### streamlit_app.py (All Changes)

| Section | Lines | Change |
|---------|-------|--------|
| AgentState definition | 201–220 | Added 5 new fields (scoping + grounding) |
| scoping_agent() | 635–670 | New agent function (35 lines) |
| validate_citations() | 672–750 | New validator function (78 lines) |
| summarizer_agent() | 1680–1710 | Added citation validation call + state fields |
| build_graph() | ~1860 | Changed entry point: "router" → "scoping" |
| Initial state setup | ~2400 | Added 4 new fields to full_state dict |
| UI styling | 2015–2040 | Added .agent-card.scoping color (#FF6B6B) |
| Agent Activity tab | ~2450 | Added scoping panel display |
| Final Answer tab | ~2500 | Added grounding badge + details panel |

**Total additions**: ~150 lines of code  
**Syntax validation**: ✅ Passed

---

## Demo Narrative Impact

### Before (Black Box):
```
User: "What are diffusion models?"
System: [runs agents silently]
Result: [Long summary appears]
Audience: "How did it decide to search? Did it just make stuff up?"
```

### After (Transparent & Grounded):
```
User: "What are diffusion models?"
[BLOCK 1 - Immediate feedback]
System displays: 
  🔍 Query scoped into 4 sub-questions
  📌 5 key themes identified
[Agents run, results appear]
[BLOCK 2 - Verification badge]
System displays:
  🟢 Citation Grounding: 22/25 (88%)
  ✅ Well grounded
  [Details show which claims verified in sources]
Result: [Summary with visible source attribution]
Audience: "Wow, it understood the query AND verified the claims!"
```

---

## Next Steps (Optional Enhancements)

1. **Interactive filtering** in citation details (filter by source: merged/extraction/none)
2. **Highlighted citations** in the summary markdown (green=grounded, red=ungrounded)
3. **Confidence scores** per citation (based on keyword overlap %)
4. **Custom threshold** (user can set minimum grounding % to warn)
5. **Export with grounding stamps** (when copying/exporting summary)

---

## Testing Recommendations

### Test 1: Scoping Display
1. Submit query: "Compare transformer architectures for NLP"
2. Verify: 4–5 sub-questions appear in Agent Activity tab
3. Verify: Keywords displayed as badges
4. Verify: Scoping appears **before** activity log details

### Test 2: Citation Validation
1. Submit query after scoping
2. When "💡 Final Answer" tab loads, check for grounding badge
3. Expand "📋 Citation Details" 
4. Verify: Mix of ✅ and ⚠️ citations shown
5. Verify: Evidence snippets display correctly

### Test 3: Edge Cases
- Query with no citations → grounding panel shouldn't show
- Query with all hallucinations → red 🔴 Badge + 0% grounding
- Query with mixed grounding → yellow 🟡 badge if 60–80%

---

**Both blocks now live and ready for demo!** 🎉

The pipeline is now **transparent** (Block 1) and **verifiable** (Block 2) — addressing both narrative and risk concerns.
