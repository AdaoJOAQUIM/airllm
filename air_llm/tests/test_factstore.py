import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from airllm.factstore import FactStore


FACTS = [
    ("Claude Shannon published A Mathematical Theory of Communication in 1948, "
     "founding information theory and defining the bit.", "shannon"),
    ("The Raspberry Pi 5 has a quad-core ARM Cortex-A76 CPU and is a popular "
     "single-board computer for local inference.", "pi"),
    ("BM25 is a ranking function based on term frequency, inverse document "
     "frequency and document length normalization.", "bm25"),
    ("Solomonoff induction weights all programs explaining the data by their "
     "brevity; it is optimal but uncomputable.", "solomonoff"),
]


class TestFactStore(unittest.TestCase):

    def make_store(self):
        store = FactStore()
        for text, source in FACTS:
            store.add(text, source=source)
        return store

    def test_search_returns_relevant_passage_first(self):
        store = self.make_store()
        cases = [
            ("who founded information theory in 1948", "shannon"),
            ("raspberry pi arm cpu", "pi"),
            ("document ranking term frequency", "bm25"),
            ("optimal but uncomputable induction", "solomonoff"),
        ]
        for query, expected_source in cases:
            results = store.search(query, k=2)
            self.assertTrue(results, f"no results for {query!r}")
            self.assertEqual(results[0]['source'], expected_source, f"query {query!r}")

    def test_retrieval_is_verbatim(self):
        store = self.make_store()
        text, source = store.get(0)
        self.assertEqual(text, FACTS[0][0])

    def test_empty_and_unknown_queries(self):
        store = self.make_store()
        self.assertEqual(store.search(""), [])
        self.assertEqual(store.search("zzzqqqxxx"), [])
        self.assertEqual(FactStore().search("anything"), [])

    def test_add_document_chunks_with_overlap(self):
        store = FactStore()
        doc = "alpha " * 300  # ~1800 chars
        ids = store.add_document(doc, source="doc", chunk_chars=1000, overlap_chars=100)
        self.assertGreaterEqual(len(ids), 2)
        text0, _ = store.get(ids[0])
        self.assertLessEqual(len(text0), 1000)

    def test_save_load_roundtrip(self):
        store = self.make_store()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'facts.bin')
            store.save(path)
            reloaded = FactStore(path)
            self.assertEqual(len(reloaded), len(FACTS))
            self.assertEqual(reloaded.get(1)[0], FACTS[1][0])
            results = reloaded.search("information theory 1948", k=1)
            self.assertEqual(results[0]['source'], "shannon")

    def test_density_report(self):
        store = self.make_store()
        report = store.density_report()
        self.assertEqual(report['passages'], len(FACTS))
        # facts on disk must beat the 2-bits/param weight equivalent
        self.assertGreater(report['density_advantage'], 1.0)
        self.assertEqual(report['params_equivalent'], report['raw_bytes'] * 8 // 2)


if __name__ == '__main__':
    unittest.main()
