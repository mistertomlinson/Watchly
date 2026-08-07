from app.services.recommendation.item_based import ItemBasedService


def test_deduplicate_candidates_preserves_first_item_order():
    items = [
        {"id": 10, "title": "First"},
        {"id": 20, "title": "Second"},
        {"id": 10, "title": "Duplicate First"},
        {"id": 30, "title": "Third"},
        {"id": 20, "title": "Duplicate Second"},
    ]

    result = ItemBasedService._deduplicate_candidates(items)

    assert [item["id"] for item in result] == [10, 20, 30]
    assert result[0]["title"] == "First"
    assert result[1]["title"] == "Second"


def test_deduplicate_candidates_supports_stremio_ids_and_preserves_missing_ids():
    unidentified_a = {"name": "No ID A"}
    unidentified_b = {"name": "No ID B"}
    items = [
        {"id": "tt123", "name": "Original"},
        unidentified_a,
        {"id": "tt123", "name": "Duplicate"},
        {"_id": "tmdb:55", "name": "Alternate ID"},
        {"_id": "tmdb:55", "name": "Duplicate Alternate"},
        unidentified_b,
    ]

    result = ItemBasedService._deduplicate_candidates(items)

    assert result == [
        {"id": "tt123", "name": "Original"},
        unidentified_a,
        {"_id": "tmdb:55", "name": "Alternate ID"},
        unidentified_b,
    ]
