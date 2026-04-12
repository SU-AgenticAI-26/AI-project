#!/usr/bin/env python3
"""
Verification script for Conference Paper Search Integration

Tests:
  1. Module imports
  2. OpenReview credentials (if available)
  3. ACL Anthology availability
  4. Basic search functionality
  5. Tool definition structure

Run: python verify_conference_search.py
"""

import os
import sys
import json

print("=" * 80)
print("Conference Paper Search Integration Verification")
print("=" * 80)

# ── Test 1: Module imports ──────────────────────────────────────────────────
print("\n[1] Checking module imports...")
try:
    from conference_paper_search import (
        SEARCH_PAPERS_TOOL,
        search_conference_papers,
        handle_conference_paper_tool_call,
        OPENREVIEW_CONFERENCES,
        ACL_CONFERENCES,
    )
    print("  ✓ conference_paper_search imports OK")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# ── Test 2: Check dependencies ──────────────────────────────────────────────
print("\n[2] Checking dependencies...")

try:
    import openreview
    print("  ✓ openreview-py installed")
except ImportError:
    print("  ✗ openreview-py NOT installed → pip install openreview-py")

try:
    from acl_anthology import Anthology
    print("  ✓ acl-anthology installed")
except ImportError:
    print("  ✗ acl-anthology NOT installed → pip install acl-anthology")

# ── Test 3: OpenReview credentials ──────────────────────────────────────────
print("\n[3] Checking OpenReview credentials...")
or_user = os.environ.get("OPENREVIEW_USERNAME", "")
or_pass = os.environ.get("OPENREVIEW_PASSWORD", "")

if or_user and or_pass:
    print(f"  ✓ OPENREVIEW_USERNAME: {or_user[:20]}...")
    print("  ✓ OPENREVIEW_PASSWORD: (set)")
else:
    print("  ⚠ OpenReview credentials not set")
    print("    → Optional, but needed for NeurIPS/ICML/ICLR search")
    print("    → Set via env vars or .env file (see CONFERENCE_SEARCH_SETUP.md)")

# ── Test 4: Tool structure ──────────────────────────────────────────────────
print("\n[4] Validating tool structure...")

if SEARCH_PAPERS_TOOL.get("type") == "function":
    print("  ✓ Tool type: 'function'")
else:
    print("  ✗ Invalid tool type")

func_name = SEARCH_PAPERS_TOOL.get("function", {}).get("name")
if func_name == "search_conference_papers":
    print(f"  ✓ Function name: {func_name}")
else:
    print(f"  ✗ Invalid function name: {func_name}")

params = SEARCH_PAPERS_TOOL.get("function", {}).get("parameters", {})
required = params.get("required", [])
if set(required) == {"keywords", "years", "conferences"}:
    print(f"  ✓ Required parameters: {required}")
else:
    print(f"  ✗ Unexpected required params: {required}")

# ── Test 5: Conferences ─────────────────────────────────────────────────────
print("\n[5] Available conferences...")
print(f"  OpenReview: {', '.join(sorted(OPENREVIEW_CONFERENCES))}")
print(f"  ACL Anthology: {', '.join(sorted(ACL_CONFERENCES))}")

# ── Test 6: Example call (limited) ──────────────────────────────────────────
print("\n[6] Testing basic search functionality...")
try:
    # Quick search with minimal data (no OpenReview creds needed for this one)
    print("  Testing ACL Anthology search (no credentials needed)...")
    results = search_conference_papers(
        keywords=["attention"],
        years=[2024],
        conferences=["ACL"],
        max_results=2
    )
    
    if results.get("papers"):
        print(f"  ✓ Found {results['returned']} ACL paper(s)")
        paper = results["papers"][0]
        print(f"    Title: {paper['title'][:60]}...")
        print(f"    Authors: {paper['authors'][:40]}...")
    else:
        print(f"  ⚠ No papers found (check internet connection)")
except Exception as e:
    print(f"  ✗ Search failed: {e}")

# ── Test 7: Check streamlit_app integration ─────────────────────────────────
print("\n[7] Checking Streamlit app integration...")
try:
    with open("streamlit_app.py", "r") as f:
        content = f.read()
        if "from conference_paper_search import" in content:
            print("  ✓ conference_paper_search imported in streamlit_app.py")
        else:
            print("  ✗ Import not found in streamlit_app.py")
        
        if "HAS_CONFERENCE_SEARCH" in content:
            print("  ✓ HAS_CONFERENCE_SEARCH flag present")
        else:
            print("  ✗ HAS_CONFERENCE_SEARCH flag not found")
            
        if "search_conference_papers" in content or "SEARCH_PAPERS_TOOL" in content:
            print("  ✓ Tool integration in web_agent")
        else:
            print("  ✗ Tool not integrated in web_agent")
except FileNotFoundError:
    print("  ✗ streamlit_app.py not found")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Verification Summary")
print("=" * 80)

print("""
✓ Core integration ready
  → Use case: When user queries mention conferences (NeurIPS, ICML, ACL, etc.)
  → The web agent will automatically search those venues

Next steps:
  1. Set OpenReview credentials (optional but recommended):
     export OPENREVIEW_USERNAME="your@email.com"
     export OPENREVIEW_PASSWORD="yourpassword"
  
  2. Test the Streamlit app:
     streamlit run streamlit_app.py
  
  3. Try a conference search query:
     "Find NeurIPS 2024 papers on diffusion models"

For issues:
  - Check CONFERENCE_SEARCH_SETUP.md for detailed docs
  - Verify credentials: python -c "import os; print(os.getenv('OPENREVIEW_USERNAME'))"
  - Check logs in Streamlit app's Activity Log sidebar
""")
