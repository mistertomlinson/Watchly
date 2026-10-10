from app.services.library_provider import make_library_item
from app.services.profile.sampling import SmartSampler
from app.services.scoring import ScoringService


def _item(rating: int | None, title: str):
    item = make_library_item(
        ids={"imdb": f"tt{1000000 + (rating or 0)}"},
        content_type="movie",
        title=title,
        year=2024,
        provider="simkl",
        watched_at="2026-08-01T00:00:00Z",
        rating=rating,
    )
    assert item is not None
    return item


def test_neutral_ratings_do_not_shape_general_taste_profile():
    neutral = _item(5, "Neutral")
    loved = _item(10, "Loved")
    sampled = SmartSampler(ScoringService()).sample_items(
        {
            "loved": [loved],
            "liked": [],
            "watched": [neutral, loved],
            "added": [neutral],
            "disliked": [],
            "removed": [],
        },
        "movie",
    )
    sampled_ids = {entry.item.id for entry in sampled}
    assert loved["_id"] in sampled_ids
    assert neutral["_id"] not in sampled_ids


def test_unrated_watched_items_keep_existing_watchly_behavior():
    watched = _item(None, "Unrated watched")
    sampled = SmartSampler(ScoringService()).sample_items(
        {
            "loved": [],
            "liked": [],
            "watched": [watched],
            "added": [],
            "disliked": [],
            "removed": [],
        },
        "movie",
    )
    assert [entry.item.id for entry in sampled] == [watched["_id"]]
