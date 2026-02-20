from app.pipeline.qdrant_store import QdrantStore


def test_build_filter_none_when_empty() -> None:
    assert QdrantStore._build_filter(None) is None
    assert QdrantStore._build_filter({}) is None


def test_build_filter_includes_all_supported_fields() -> None:
    qfilter = QdrantStore._build_filter(
        {
            "filename": "training",
            "tags": ["ml", "onboarding"],
            "date_from": "2026-01-01T00:00:00Z",
            "date_to": "2026-02-01T00:00:00Z",
        }
    )

    assert qfilter is not None
    must = qfilter.must or []
    keys = [c.key for c in must]

    assert keys.count("filename") == 1
    assert keys.count("tags") == 2
    assert keys.count("date_indexed") == 2
