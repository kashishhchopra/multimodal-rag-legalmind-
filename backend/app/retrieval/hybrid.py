"""Lexical (BM25) retrieval and reciprocal-rank fusion.

Legal queries hinge on exact tokens (section numbers, party names, citations) that dense
embeddings blur, so we fuse a BM25 ranking with the dense ranking using RRF:
score(d) = sum over rankers of 1 / (rrf_k + rank). Pure Python, no extra dependency.

Scale note: the BM25 index is built in memory from the Chroma text collection and cached
until its size changes. That is fine for Tier 1 (hundreds of thousands of chunks is the
limit); Tier 2 should move lexical search to Qdrant sparse vectors / OpenSearch.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

# Keep section refs like "318(4)", "65B", "138" as single tokens alongside plain words.
_TOKEN = re.compile(r"\d+[a-z]?(?:\(\w+\))*|[a-z][a-z'\-]+", re.IGNORECASE)
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were which with".split()
)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text) if t.lower() not in _STOP]


class BM25Index:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(t)) for t in texts]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._tf)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._postings: Dict[str, List[int]] = {}
        for i, tf in enumerate(self._tf):
            for t in tf:
                self._postings.setdefault(t, []).append(i)

    def search(self, query: str, k: int) -> List[tuple[str, float]]:
        scores: Dict[int, float] = {}
        for t in set(tokenize(query)):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i in self._postings[t]:
                tf = self._tf[i][t]
                denom = tf + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avg or 1.0))
                scores[i] = scores.get(i, 0.0) + idf * tf * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [(self.ids[i], s) for i, s in top]


def rrf_fuse(rankings: Iterable[Sequence[str]], rrf_k: int = 60) -> List[tuple[str, float]]:
    """Reciprocal rank fusion over several ranked id lists; returns (id, fused score) best-first."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking, start=1):
            fused[_id] = fused.get(_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
