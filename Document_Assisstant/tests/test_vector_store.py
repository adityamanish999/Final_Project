import pytest

from core import config, vector_store
from tests.conftest import make_chunks


def test_add_document_and_search_roundtrip(fake_store):
    added = vector_store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    assert added == 3
    assert vector_store.is_ingested("hash1")

    results = vector_store.search("chunk 0 from doc1.pdf", top_k=2)
    assert len(results) == 2
    assert results[0]["source"] == "doc1.pdf"


def test_add_document_is_idempotent_for_same_hash(fake_store):
    vector_store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    added_again = vector_store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")

    assert added_again == 0
    assert vector_store.store["index"].ntotal == 3


def test_save_and_load_persists(fake_store):
    vector_store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    vector_store.save_store()

    vector_store.load_store(fake_store)  # reload from disk
    assert vector_store.store["index"].ntotal == 3
    assert vector_store.is_ingested("hash1")
    assert vector_store.list_documents()[0]["filename"] == "doc1.pdf"


def test_embedding_model_mismatch_raises(fake_store, monkeypatch):
    vector_store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    vector_store.save_store()

    monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "different-model")
    with pytest.raises(ValueError):
        vector_store.load_store(fake_store)


def test_clear_resets_store(fake_store):
    vector_store.add_document(make_chunks("doc1.pdf"), file_hash="hash1", filename="doc1.pdf")
    vector_store.save_store()

    vector_store.clear_store()
    assert vector_store.store["index"].ntotal == 0
    assert vector_store.list_documents() == []
    assert not vector_store.is_ingested("hash1")
