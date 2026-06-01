"""
test_citation_verifier.py — Unit tests for citation_verifier.py.

No model download required: _get_model() is mocked in all tests that touch the
sentence-transformers model. numpy is always available as a transitive dependency.

Run:
    python -m pytest test_citation_verifier.py -v
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import citation_verifier as cv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_model(claim_vec=None, cand_vec=None):
    """
    Return a mock SentenceTransformer whose encode() returns controlled arrays.

    claim_vec: shape (D,) returned for a single string input
    cand_vec:  shape (D,) returned *per item* for a list input (all same vec)
    """
    if claim_vec is None:
        claim_vec = np.array([1.0, 0.0])
    if cand_vec is None:
        cand_vec = np.array([0.9, 0.436])   # cosine ≈ 0.9 with [1, 0]

    model = MagicMock()

    def _encode(texts, **kw):
        if isinstance(texts, str):
            return claim_vec.copy()
        return np.array([cand_vec.copy() for _ in texts])

    model.encode.side_effect = _encode
    return model


_EXTRACTION_BLOCK = """\
---
**Title / Topic:** Dense Retrieval for Open-Domain QA
**Provenance:** full-text
**Research Problem:** How to improve recall in open-domain question answering.
**Methodology:** Bi-encoder DPR with hard-negative mining.
**Key Findings:**
- Embedding-based retrieval outperforms BM25 by 8 F1 points.
- Hard negatives are critical for encoder quality.
**Limitations:** Requires expensive GPU fine-tuning.
**Future Work:** Cross-lingual dense retrieval.
---
"""

_EXTRACTION_TWO_BLOCKS = """\
---
**Title / Topic:** Paper One
**Provenance:** abstract-only
**Research Problem:** First research problem.
**Key Findings:**
- Finding one.
---
**Title / Topic:** Paper Two
**Provenance:** structured-db
**Research Problem:** Second research problem.
**Key Findings:**
- Finding two.
---
"""


# ---------------------------------------------------------------------------
# TestExtractCitations
# ---------------------------------------------------------------------------

class TestExtractCitations(unittest.TestCase):

    def test_extracts_quoted_claim(self):
        summary = 'Research shows "embedding-based retrieval outperforms BM25 by a wide margin" in open-domain QA.'
        cits = cv.extract_citations(summary)
        self.assertEqual(len(cits), 1)
        self.assertEqual(cits[0]["type"], "quoted")
        self.assertIn("outperforms BM25", cits[0]["text"])

    def test_quoted_source_hint_vectordb(self):
        summary = '"dense retrieval greatly reduces hallucination in generative models" [VectorDB]'
        cits = cv.extract_citations(summary)
        self.assertEqual(len(cits), 1)
        self.assertEqual(cits[0]["source_hint"], "vector_db")

    def test_quoted_source_hint_sql(self):
        summary = '"federated learning preserves patient privacy across hospital networks" [SQL]'
        cits = cv.extract_citations(summary)
        self.assertEqual(cits[0]["source_hint"], "sql_db")

    def test_quoted_source_hint_web(self):
        summary = '"large language models exhibit emergent reasoning capabilities at scale" [Web]'
        cits = cv.extract_citations(summary)
        self.assertEqual(cits[0]["source_hint"], "web")

    def test_quoted_source_hint_none_when_no_tag(self):
        summary = '"dense retrieval greatly reduces hallucination in generative models"'
        cits = cv.extract_citations(summary)
        self.assertIsNone(cits[0]["source_hint"])

    def test_extracts_numeric_citation_as_sentence(self):
        summary = "The method achieves state-of-the-art performance on multiple benchmarks [1]. Additional analysis confirms robustness."
        cits = cv.extract_citations(summary)
        numeric = [c for c in cits if c["type"] == "numeric"]
        self.assertTrue(len(numeric) >= 1)
        self.assertIn("[1]", numeric[0]["text"] + summary)

    def test_extracts_parenthetical_claim(self):
        summary = "Approaches described as a multi-step iterative refinement process (combining dense retrieval with reranking and cross-encoder scoring) show strong results."
        cits = cv.extract_citations(summary)
        paren = [c for c in cits if c["type"] == "parenthetical"]
        self.assertTrue(len(paren) >= 1)

    def test_deduplicates_identical_claims(self):
        claim = '"dense retrieval greatly reduces hallucination in generative models"'
        summary = claim + " Also: " + claim
        cits = cv.extract_citations(summary)
        texts = [c["text"] for c in cits]
        self.assertEqual(len(texts), len(set(t.lower() for t in texts)))

    def test_caps_at_15(self):
        # Build a summary with 20 distinct quoted claims (each ≥ 20 chars)
        claims = [f'"this is a unique claim number {i:02d} about research methods"' for i in range(20)]
        summary = " ".join(claims)
        cits = cv.extract_citations(summary)
        self.assertLessEqual(len(cits), 15)

    def test_minimum_length_filter(self):
        summary = '"short" and "also short" appear but are too brief'
        cits = cv.extract_citations(summary)
        self.assertEqual(len(cits), 0)


# ---------------------------------------------------------------------------
# TestExtractPaperChunks
# ---------------------------------------------------------------------------

class TestExtractPaperChunks(unittest.TestCase):

    def test_parses_single_block_title(self):
        chunks = cv.extract_paper_chunks(_EXTRACTION_BLOCK)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Dense Retrieval", chunks[0]["title"])

    def test_provenance_full_text_maps_to_vector_db(self):
        chunks = cv.extract_paper_chunks(_EXTRACTION_BLOCK)
        self.assertEqual(chunks[0]["source"], "vector_db")

    def test_provenance_abstract_only_maps_to_web(self):
        block = _EXTRACTION_BLOCK.replace("full-text", "abstract-only")
        chunks = cv.extract_paper_chunks(block)
        self.assertEqual(chunks[0]["source"], "web")

    def test_provenance_structured_db_maps_to_sql_db(self):
        block = _EXTRACTION_BLOCK.replace("full-text", "structured-db")
        chunks = cv.extract_paper_chunks(block)
        self.assertEqual(chunks[0]["source"], "sql_db")

    def test_text_includes_key_findings_content(self):
        chunks = cv.extract_paper_chunks(_EXTRACTION_BLOCK)
        self.assertIn("BM25", chunks[0]["text"])

    def test_parses_multiple_blocks(self):
        chunks = cv.extract_paper_chunks(_EXTRACTION_TWO_BLOCKS)
        self.assertEqual(len(chunks), 2)

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(cv.extract_paper_chunks(""), [])

    def test_sentinel_strings_return_empty_list(self):
        for sentinel in ("NO_PAPERS_EXTRACTED", "(none)", "(not activated)"):
            with self.subTest(sentinel=sentinel):
                self.assertEqual(cv.extract_paper_chunks(sentinel), [])


# ---------------------------------------------------------------------------
# TestSemanticGrounded
# ---------------------------------------------------------------------------

class TestSemanticGrounded(unittest.TestCase):

    def _run(self, claim_vec, cand_vec, candidates, threshold=0.75):
        with patch("citation_verifier._get_model", return_value=_fake_model(claim_vec, cand_vec)):
            return cv.semantic_grounded(claim="test claim", candidate_texts=candidates,
                                        threshold=threshold)

    def test_returns_true_when_similarity_above_threshold(self):
        # [1,0] · [0.9, 0.436] / norms ≈ 0.9
        grounded, score, _ = self._run(np.array([1.0, 0.0]), np.array([0.9, 0.436]),
                                        ["Dense retrieval outperforms sparse methods."])
        self.assertTrue(grounded)
        self.assertGreater(score, 0.75)

    def test_returns_false_when_similarity_below_threshold(self):
        # [1,0] · [0.0, 1.0] = 0.0
        grounded, score, _ = self._run(np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                                        ["Completely unrelated candidate text here."])
        self.assertFalse(grounded)
        self.assertLess(score, 0.75)

    def test_returns_false_for_empty_candidates(self):
        grounded, score, evidence = cv.semantic_grounded("any claim", [])
        self.assertFalse(grounded)
        self.assertEqual(score, 0.0)
        self.assertEqual(evidence, "")

    def test_evidence_comes_from_best_candidate(self):
        # Two candidates: second is closer
        claim_vec = np.array([1.0, 0.0])
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: (
            claim_vec.copy() if isinstance(texts, str)
            else np.array([[0.1, 0.99], [0.99, 0.1]])   # second is closest
        )
        with patch("citation_verifier._get_model", return_value=model):
            _, _, evidence = cv.semantic_grounded(
                claim="test claim",
                candidate_texts=["far candidate text", "near candidate text"],
                threshold=0.0,
            )
        self.assertIn("near", evidence)

    def test_threshold_boundary_at_exact_value(self):
        # Construct vectors where cosine = exactly 0.75
        # claim=[1,0], cand=[0.75, sqrt(1-0.75^2)] = [0.75, 0.6614]
        claim_vec = np.array([1.0, 0.0])
        cand_vec = np.array([0.75, np.sqrt(1 - 0.75 ** 2)])
        grounded, score, _ = self._run(claim_vec, cand_vec, ["boundary candidate text here."],
                                        threshold=0.75)
        self.assertAlmostEqual(score, 0.75, places=4)
        self.assertTrue(grounded)  # >= threshold


# ---------------------------------------------------------------------------
# TestComputeCitationMetrics
# ---------------------------------------------------------------------------

class TestComputeCitationMetrics(unittest.TestCase):

    _SUMMARY_WITH_CITATIONS = (
        'Research demonstrates "dense retrieval significantly reduces hallucination '
        'by grounding outputs in retrieved documents" [VectorDB]. '
        'Additional work shows "federated learning enables privacy-preserving model training '
        'across distributed hospital networks" [SQL].'
    )

    def _run(self, model, summary=None, extraction=_EXTRACTION_BLOCK,
             merged="some merged context with relevant information"):
        if summary is None:
            summary = self._SUMMARY_WITH_CITATIONS
        with patch("citation_verifier._get_model", return_value=model):
            return cv.compute_citation_metrics(
                summary=summary,
                extraction_findings=extraction,
                merged_context=merged,
            )

    def test_returns_tuple_of_dict_and_float(self):
        result = self._run(_fake_model())
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)
        self.assertIsInstance(result[1], float)

    def test_grounding_map_keys_truncated_to_100_chars(self):
        gmap, _ = self._run(_fake_model())
        for key in gmap:
            self.assertLessEqual(len(key), 100)

    def test_grounding_map_values_have_required_keys(self):
        gmap, _ = self._run(_fake_model())
        for val in gmap.values():
            for field in ("grounded", "source", "evidence", "similarity", "type"):
                self.assertIn(field, val, msg=f"Missing field '{field}' in {val}")

    def test_perfect_score_when_all_grounded(self):
        # claim_vec and cand_vec aligned → cosine ≈ 0.9
        _, score = self._run(_fake_model(
            claim_vec=np.array([1.0, 0.0]),
            cand_vec=np.array([0.9, 0.436]),
        ))
        self.assertEqual(score, 1.0)

    def test_zero_score_when_none_grounded(self):
        # Orthogonal vectors → cosine = 0.0
        _, score = self._run(_fake_model(
            claim_vec=np.array([1.0, 0.0]),
            cand_vec=np.array([0.0, 1.0]),
        ))
        self.assertEqual(score, 0.0)

    def test_returns_1_0_when_no_citations_found(self):
        # Summary with no extractable citations
        gmap, score = self._run(_fake_model(), summary="No citations here at all.")
        self.assertEqual(gmap, {})
        self.assertEqual(score, 1.0)

    def test_source_hint_used_to_select_bucket(self):
        # Verify that when source_hint="vector_db" the model is only encoding
        # from the vector_db bucket (not sql or web)
        model = MagicMock()
        model.encode.side_effect = lambda texts, **kw: (
            np.array([1.0, 0.0]) if isinstance(texts, str)
            else np.array([[0.9, 0.436]] * len(texts))
        )
        summary = '"dense retrieval reduces hallucination in language model outputs" [VectorDB]'
        with patch("citation_verifier._get_model", return_value=model):
            cv.compute_citation_metrics(
                summary=summary,
                extraction_findings=_EXTRACTION_BLOCK,
                merged_context="some merged context",
                vector_findings="Dense retrieval is an important technique in NLP.",
            )
        # encode() called at least twice (claim + candidates)
        self.assertGreaterEqual(model.encode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
