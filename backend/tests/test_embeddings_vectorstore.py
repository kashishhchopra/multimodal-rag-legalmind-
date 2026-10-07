"""Tests for the embedding layer (fake mode) and the ChromaDB vector store."""
from __future__ import annotations

import numpy as np

from app.models import embeddings
from app.retrieval import vector_store


def test_text_embeddings_shape_and_norm():
    vecs = embeddings.embed_text_passages(["hello world", "circuit resistor Q1"])
    assert vecs.shape == (2, embeddings.text_dim())
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_text_query_similarity_orders_sensibly():
    passages = embeddings.embed_text_passages(
        ["R1 is a current limiting resistor", "the weather is sunny today"]
    )
    q = embeddings.embed_text_query("what does the resistor R1 do")
    sims = passages @ q
    assert sims[0] > sims[1]  # lexical overlap wins even in fake mode


def test_image_embeddings_shape():
    from PIL import Image

    imgs = [Image.new("RGB", (32, 32), "white"), Image.new("RGB", (32, 32), "black")]
    vecs = embeddings.embed_images(imgs)
    assert vecs.shape == (2, embeddings.image_dim())


def test_vector_store_add_query_get_delete():
    doc_id = "vs_unit"
    ids = [f"{doc_id}_c{i}" for i in range(3)]
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    vecs = embeddings.embed_text_passages(texts)
    metas = [{"document_id": doc_id, "page_number": i + 1, "chunk_id": ids[i]} for i in range(3)]
    vector_store.add_text_chunks(ids, vecs, texts, metas)

    q = embeddings.embed_text_query("gamma delta")
    hits = vector_store.query_text(q, k=2, document_id=doc_id)
    assert hits
    assert hits[0]["id"] == ids[1]
    assert 0.0 <= hits[0]["score"] <= 1.0

    got = vector_store.get_by_id(vector_store.TEXT_COLLECTION, ids[0])
    assert got and got["document"] == "alpha beta"

    vector_store.delete_document(doc_id)
    assert vector_store.query_text(q, k=2, document_id=doc_id) == []


def test_query_filters_by_document_id():
    v = embeddings.embed_text_passages(["shared token"])
    vector_store.add_text_chunks(["docA_c0"], v, ["shared token"], [{"document_id": "docA", "page_number": 1}])
    vector_store.add_text_chunks(["docB_c0"], v, ["shared token"], [{"document_id": "docB", "page_number": 1}])
    q = embeddings.embed_text_query("shared token")
    hits_a = vector_store.query_text(q, k=5, document_id="docA")
    assert all(h["metadata"]["document_id"] == "docA" for h in hits_a)
    vector_store.delete_document("docA")
    vector_store.delete_document("docB")
