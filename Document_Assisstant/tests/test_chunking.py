from core.chunking import ChunkConfig, chunk_document, chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = "a" * 1000
    config = ChunkConfig(chunk_size=100, overlap=20)
    chunks = chunk_text(text, page=1, source="doc.pdf", chunk_config=config)

    assert all(len(c["text"]) <= 100 for c in chunks)
    # step = 80, so 1000 chars need ceil(1000/80) = 13 windows (last one shorter)
    assert len(chunks) == 13
    assert all(c["page"] == 1 and c["source"] == "doc.pdf" for c in chunks)


def test_chunk_text_empty_string_produces_no_chunks():
    config = ChunkConfig(chunk_size=100, overlap=20)
    assert chunk_text("", page=1, source="doc.pdf", chunk_config=config) == []


def test_chunk_document_does_not_cross_page_boundaries():
    pages = [
        {"page": 1, "text": "x" * 50, "source": "doc.pdf"},
        {"page": 2, "text": "y" * 50, "source": "doc.pdf"},
    ]
    config = ChunkConfig(chunk_size=100, overlap=10)
    chunks = chunk_document(pages, config)

    assert len(chunks) == 2
    assert chunks[0]["page"] == 1 and set(chunks[0]["text"]) == {"x"}
    assert chunks[1]["page"] == 2 and set(chunks[1]["text"]) == {"y"}


def test_unsupported_strategy_raises():
    import pytest

    config = ChunkConfig(chunk_size=100, overlap=10, strategy="semantic")
    with pytest.raises(ValueError):
        chunk_text("some text", page=1, source="doc.pdf", chunk_config=config)
