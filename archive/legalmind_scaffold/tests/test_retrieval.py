from src.retrieval.hybrid_retriever import RetrievedChunk, _reciprocal_rank_fusion
from src.schemas import Chunk, ChunkMetadata


class _FakePoint:
    def __init__(self, point_id: str, payload: dict):
        self.id = point_id
        self.payload = payload


def _payload(doc_id: str, page: int) -> dict:
    return {
        "text": f"chunk from {doc_id}",
        "metadata": ChunkMetadata(doc_id=doc_id, page=page).model_dump(),
    }


def test_reciprocal_rank_fusion_favors_chunks_in_both_lists():
    dense_hits = [_FakePoint("a", _payload("doc1", 1)), _FakePoint("b", _payload("doc1", 2))]
    sparse_hits = [_FakePoint("b", _payload("doc1", 2)), _FakePoint("c", _payload("doc1", 3))]

    fused = _reciprocal_rank_fusion([dense_hits, sparse_hits])

    assert fused[0].chunk.chunk_id == "b"  # appears in both lists, should rank first
    ids = [r.chunk.chunk_id for r in fused]
    assert set(ids) == {"a", "b", "c"}
