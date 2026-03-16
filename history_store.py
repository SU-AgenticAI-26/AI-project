"""
history_store.py — Per-user run history stored as markdown files.

Directory layout:
    history/
        {username}/
            {YYYYMMDD_HHMMSS}_{slug}.md     ← one file per completed run
            index.json                       ← fast metadata index (rebuilt on load)

Each markdown file has a YAML-ish frontmatter block between --- delimiters,
followed by the full literature review content. This lets the system parse
metadata cheaply and agents inject relevant prior context.

Public API
──────────
    store = HistoryStore(username)
    store.save(state)               → str  (filepath saved)
    store.list_runs()               → list[RunMeta]
    store.load_run(filename)        → dict  (parsed: meta + content sections)
    store.search_runs(query, n=3)   → list[RunMeta]  (keyword match on query/themes)
    store.prior_context(query, n=2) → str  (markdown block for agent injection)
"""

from __future__ import annotations
import json
import os
import re
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


HISTORY_ROOT = Path("history")


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class RunMeta:
    filename:    str
    username:    str
    query:       str
    timestamp:   str          # ISO-8601
    n_papers:    int
    n_steps:     int
    themes:      list[str]
    approved:    bool
    slug:        str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _slugify(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return s[:maxlen].strip("-")


def _render_md(state: dict, username: str, timestamp: str) -> str:
    """Build the full markdown document for a completed run."""
    query         = state.get("query", "")
    sub_questions = state.get("sub_questions", [])
    summaries     = state.get("paper_summaries", [])
    synthesis     = state.get("synthesis", "")
    gaps          = state.get("gaps", "")
    plan_prose    = state.get("research_plan", "")
    research_steps = state.get("research_steps", [])
    risks         = state.get("risks_and_mitigations", "")
    themes        = state.get("themes", [])
    contradictions = state.get("contradictions", [])
    uncovered     = state.get("uncovered_sub_questions", [])
    val_result    = state.get("validation_result", {})
    approved      = val_result.get("approved", True)
    sources_sel   = state.get("sources_selected", [])

    theme_names   = [t.get("theme_name", t) if isinstance(t, dict) else str(t)
                     for t in themes]

    # ── Frontmatter ────────────────────────────────────────────────────────
    fm_themes = json.dumps(theme_names)
    fm = textwrap.dedent(f"""\
        ---
        query: {json.dumps(query)}
        username: {json.dumps(username)}
        timestamp: {timestamp}
        n_papers: {len(summaries)}
        n_steps: {len(research_steps)}
        themes: {fm_themes}
        approved: {str(approved).lower()}
        sources: {json.dumps(sources_sel)}
        ---
    """)

    # ── Body ───────────────────────────────────────────────────────────────
    parts = [fm, f"# {query}\n"]

    if sub_questions:
        parts.append("## Sub-questions\n")
        for i, q in enumerate(sub_questions, 1):
            parts.append(f"{i}. {q}")
        parts.append("")

    if synthesis:
        parts.append("## Thematic Synthesis\n")
        parts.append(synthesis)
        parts.append("")

    if contradictions or uncovered:
        parts.append("## Issues\n")
        for c in contradictions:
            parts.append(f"- **Contradiction:** {c}")
        for u in uncovered:
            parts.append(f"- **Insufficient coverage:** {u}")
        parts.append("")

    if gaps:
        parts.append("## Research Gaps\n")
        parts.append(gaps)
        parts.append("")

    if plan_prose:
        parts.append("## Research Plan\n")
        parts.append(plan_prose)
        parts.append("")

    if research_steps:
        parts.append("## Research Steps\n")
        for s in research_steps:
            n     = s.get("step", "")
            title = s.get("title", "")
            desc  = s.get("description", "")
            gps   = s.get("grounding_papers", [])
            fw    = s.get("future_work_link", "")
            parts.append(f"### Step {n}: {title}\n")
            parts.append(desc)
            if gps:
                parts.append(f"\n*Grounded in:* {', '.join(gps)}")
            if fw:
                parts.append(f"*Future work link:* {fw}")
            parts.append("")

    if risks:
        parts.append("## Methodological Risks\n")
        parts.append(risks)
        parts.append("")

    if summaries:
        parts.append("## Papers\n")
        for i, p in enumerate(summaries, 1):
            url   = p.get("url", "")
            title = p.get("title", "")
            year  = p.get("year", "?")
            src   = p.get("source", "")
            cites = p.get("citation_count", 0)
            link  = f"[{title}]({url})" if url else title
            parts.append(f"{i}. {link} ({year}) [{src}] — {cites} citations")
        parts.append("")

    return "\n".join(parts)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Extract YAML-ish frontmatter from between --- delimiters.
    Returns (meta_dict, body_text).
    """
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return {}, text
    fm_block = m.group(1)
    body     = text[m.end():]
    meta: dict = {}
    for line in fm_block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        try:
            meta[k.strip()] = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            meta[k.strip()] = v
    return meta, body


# ── Main class ─────────────────────────────────────────────────────────────────

class HistoryStore:
    def __init__(self, username: str):
        self.username = username.strip().lower() or "default"
        self.user_dir = HISTORY_ROOT / self.username
        self.user_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──────────────────────────────────────────────────────────────

    def save(self, state: dict) -> str:
        """Persist a completed run. Returns the filename (not full path)."""
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(state.get("query", "query"))
        name = f"{ts}_{slug}.md"
        path = self.user_dir / name
        content = _render_md(state, self.username, datetime.now().isoformat())
        path.write_text(content, encoding="utf-8")
        self._update_index()
        return name

    # ── List ──────────────────────────────────────────────────────────────

    def list_runs(self) -> list[RunMeta]:
        """Return RunMeta for all saved runs, newest first."""
        runs = []
        for f in sorted(self.user_dir.glob("*.md"), reverse=True):
            meta = self._meta_for(f)
            if meta:
                runs.append(meta)
        return runs

    # ── Load ──────────────────────────────────────────────────────────────

    def load_run(self, filename: str) -> dict:
        """
        Parse a saved run file.
        Returns {"meta": RunMeta, "body": str, "raw": str}.
        """
        path = self.user_dir / filename
        if not path.exists():
            return {}
        raw  = path.read_text(encoding="utf-8")
        meta_dict, body = _parse_frontmatter(raw)
        meta = RunMeta(
            filename  = filename,
            username  = meta_dict.get("username", self.username),
            query     = meta_dict.get("query", ""),
            timestamp = meta_dict.get("timestamp", ""),
            n_papers  = int(meta_dict.get("n_papers", 0)),
            n_steps   = int(meta_dict.get("n_steps", 0)),
            themes    = meta_dict.get("themes", []),
            approved  = meta_dict.get("approved", True),
            slug      = filename,
        )
        return {"meta": meta, "body": body, "raw": raw}

    # ── Search ────────────────────────────────────────────────────────────

    def search_runs(self, query: str, n: int = 5) -> list[RunMeta]:
        """
        Keyword search across saved run queries and theme labels.
        Returns up to n most-relevant RunMeta objects, newest first within rank.
        """
        keywords = set(re.findall(r"[a-z]{4,}", query.lower()))
        scored: list[tuple[int, RunMeta]] = []
        for run in self.list_runs():
            target = run.query.lower() + " " + " ".join(run.themes).lower()
            score  = sum(1 for kw in keywords if kw in target)
            if score > 0:
                scored.append((score, run))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:n]]

    # ── Prior context for agent injection ─────────────────────────────────

    def prior_context(self, query: str, n: int = 2) -> str:
        """
        Return a markdown block summarising the n most relevant prior runs.
        Designed for injection into the Orchestrator or ScopingAgent prompts.
        Empty string if no relevant history.
        """
        relevant = self.search_runs(query, n=n)
        if not relevant:
            return ""
        lines = ["## Relevant prior research (from your history)\n"]
        for r in relevant:
            ts_display = r.timestamp[:10] if r.timestamp else ""
            lines.append(f"### {r.query} ({ts_display})")
            lines.append(f"- Papers: {r.n_papers}  |  Themes: {', '.join(r.themes) or 'n/a'}")
            lines.append(f"- Validation: {'approved' if r.approved else 'not fully approved'}")
            lines.append(f"- File: `{r.filename}`\n")
        lines.append(
            "_You may build on, contrast with, or extend these prior results "
            "rather than repeating the same ground._\n"
        )
        return "\n".join(lines)

    # ── Index ─────────────────────────────────────────────────────────────

    def _meta_for(self, path: Path) -> Optional[RunMeta]:
        try:
            raw  = path.read_text(encoding="utf-8")
            meta_dict, _ = _parse_frontmatter(raw)
            if not meta_dict.get("query"):
                return None
            return RunMeta(
                filename  = path.name,
                username  = meta_dict.get("username", self.username),
                query     = meta_dict.get("query", ""),
                timestamp = meta_dict.get("timestamp", ""),
                n_papers  = int(meta_dict.get("n_papers", 0)),
                n_steps   = int(meta_dict.get("n_steps", 0)),
                themes    = meta_dict.get("themes", []),
                approved  = meta_dict.get("approved", True),
                slug      = path.stem,
            )
        except Exception:
            return None

    def _update_index(self):
        """Write a lightweight JSON index of all runs for fast listing."""
        runs = self.list_runs()
        idx  = [asdict(r) for r in runs]
        (self.user_dir / "index.json").write_text(
            json.dumps(idx, indent=2), encoding="utf-8"
        )
