from app.services.library_provider import (
    RATING_POLICY,
    RatingBucket,
    add_rated_item,
    empty_library,
    make_library_item,
)


def test_rating_buckets_match_watchly_policy():
    assert RATING_POLICY.classify(10) == RatingBucket.LOVED
    assert all(RATING_POLICY.classify(value) == RatingBucket.LIKED for value in (7, 8, 9))
    assert all(RATING_POLICY.classify(value) == RatingBucket.NEUTRAL for value in (3, 4, 5, 6))
    assert all(RATING_POLICY.classify(value) == RatingBucket.DISLIKED for value in (1, 2))


def test_only_positive_ratings_receive_taste_weight():
    assert RATING_POLICY.taste_weight(10) == 1.0
    assert RATING_POLICY.taste_weight(7) == 0.65
    assert RATING_POLICY.taste_weight(9) == 0.65
    assert RATING_POLICY.taste_weight(6) == 0.0
    assert RATING_POLICY.taste_weight(2) == 0.0


def test_neutral_rating_remains_watched_only():
    library = empty_library()
    item = make_library_item(
        ids={"imdb": "tt1234567"},
        content_type="movie",
        title="Neutral Movie",
        year=2024,
        provider="simkl",
        watched_at="2026-08-01T00:00:00Z",
        rating=5,
    )
    assert item is not None
    library["watched"].append(item)
    add_rated_item(library, item)

    assert [entry["_id"] for entry in library["watched"]] == ["tt1234567"]
    assert library["loved"] == []
    assert library["liked"] == []
    assert library["disliked"] == []


def test_disliked_item_is_never_positive_anchor():
    library = empty_library()
    item = make_library_item(
        ids={"tmdb": 42},
        content_type="series",
        title="Disliked Show",
        year=2020,
        provider="trakt",
        rating=1,
    )
    assert item is not None
    add_rated_item(library, item)

    assert item["_taste_weight"] == 0.0
    assert item["_is_disliked"] is True
    assert library["disliked"] == [item]
    assert library["loved"] == []
    assert library["liked"] == []
