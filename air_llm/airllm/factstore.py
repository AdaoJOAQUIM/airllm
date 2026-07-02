"""
Local, offline fact store: lossless compressed passages + BM25 retrieval.

Rationale (docs/THEORY.md, Theorem 2): weights store ~2 bits of knowledge
per parameter (~1 bit per fp16 byte), while compressed text stores ~8 bits
per byte -- facts are ~an order of magnitude denser on disk than in
parameters, and retrieval returns them verbatim (lossless). The weights
should hold the *reasoner*; the facts should live here.

Pure standard library: zlib for storage, BM25 (Robertson-Sparck Jones)
for ranking. No network, no external services.
"""

import json
import math
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path


_WORD_RE = re.compile(r"[a-z0-9]+")

# BM25 constants (standard values from the literature)
_K1 = 1.5
_B = 0.75


def _tokenize(text):
    return _WORD_RE.findall(text.lower())


class FactStore:
    """
    Append-only store of text passages with BM25 search.

    Passages are zlib-compressed on disk; the inverted index is rebuilt
    from the passages at load time (the store is the single source of
    truth -- no index/state divergence possible).
    """

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._passages = []          # list of (compressed bytes, source)
        self._doc_tokens = []        # list of Counter per passage
        self._doc_len = []
        self._df = defaultdict(int)  # term -> number of passages containing it
        if self.path is not None and self.path.exists():
            self._load()

    def __len__(self):
        return len(self._passages)

    def add(self, text, source=None):
        """Add one passage. Returns its id."""
        tokens = Counter(_tokenize(text))
        self._passages.append((zlib.compress(text.encode('utf-8'), 6), source))
        self._doc_tokens.append(tokens)
        self._doc_len.append(sum(tokens.values()))
        for term in tokens:
            self._df[term] += 1
        return len(self._passages) - 1

    def add_document(self, text, source=None, chunk_chars=1000, overlap_chars=100):
        """Split a document into overlapping passages and add them all."""
        ids = []
        step = max(1, chunk_chars - overlap_chars)
        for start in range(0, max(1, len(text)), step):
            chunk = text[start:start + chunk_chars]
            if chunk.strip():
                ids.append(self.add(chunk, source=source))
            if start + chunk_chars >= len(text):
                break
        return ids

    def get(self, passage_id):
        compressed, source = self._passages[passage_id]
        return zlib.decompress(compressed).decode('utf-8'), source

    def search(self, query, k=5):
        """BM25 top-k. Returns list of dicts: {id, score, text, source}."""
        query_terms = _tokenize(query)
        n = len(self._passages)
        if n == 0 or not query_terms:
            return []
        avg_len = sum(self._doc_len) / n

        scores = defaultdict(float)
        for term in query_terms:
            df = self._df.get(term)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tokens in enumerate(self._doc_tokens):
                tf = tokens.get(term)
                if not tf:
                    continue
                norm = _K1 * (1 - _B + _B * self._doc_len[doc_id] / avg_len)
                scores[doc_id] += idf * tf * (_K1 + 1) / (tf + norm)

        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results = []
        for doc_id, score in top:
            text, source = self.get(doc_id)
            results.append({'id': doc_id, 'score': score, 'text': text, 'source': source})
        return results

    def save(self, path=None):
        path = Path(path) if path is not None else self.path
        if path is None:
            raise ValueError("no path given")
        records = [{'z': compressed.hex(), 's': source}
                   for compressed, source in self._passages]
        payload = zlib.compress(json.dumps(records).encode('utf-8'), 6)
        path.write_bytes(payload)
        self.path = path

    def _load(self):
        records = json.loads(zlib.decompress(self.path.read_bytes()).decode('utf-8'))
        for record in records:
            compressed, source = bytes.fromhex(record['z']), record['s']
            text = zlib.decompress(compressed).decode('utf-8')
            tokens = Counter(_tokenize(text))
            self._passages.append((compressed, source))
            self._doc_tokens.append(tokens)
            self._doc_len.append(sum(tokens.values()))
            for term in tokens:
                self._df[term] += 1

    def density_report(self):
        """
        How much knowledge sits here vs. what parameters would need.

        Under the ~2 bits/param capacity law (docs/THEORY.md, Theorem 2),
        storing these bytes in weights would take raw_bytes*8/2 parameters
        -- i.e. params_equivalent * 2 bytes of fp16 on disk.
        """
        raw = sum(len(zlib.decompress(c)) for c, _ in self._passages)
        stored = sum(len(c) for c, _ in self._passages)
        params_equivalent = raw * 8 // 2
        return {
            'passages': len(self._passages),
            'raw_bytes': raw,
            'stored_bytes': stored,
            'params_equivalent': params_equivalent,
            'fp16_bytes_equivalent': params_equivalent * 2,
            'density_advantage': (params_equivalent * 2 / stored) if stored else 0.0,
        }
