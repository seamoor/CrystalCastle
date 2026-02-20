from app.pipeline.chunking import chunk_text


def test_chunk_text_overlap_and_order() -> None:
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)

    assert len(chunks) == 3
    assert chunks[0].index == 0
    assert chunks[1].index == 1
    assert chunks[2].index == 2
    assert len(chunks[0].text) == 40
    assert len(chunks[1].text) == 40
    assert len(chunks[2].text) == 40


def test_chunk_text_empty_input() -> None:
    assert chunk_text("   \n\t ", chunk_size=100, chunk_overlap=20) == []
