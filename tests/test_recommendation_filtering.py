import asyncio

from app.services.recommendation.filtering import RecommendationFiltering


def test_exclusion_sets_use_preserved_external_tmdb_id():
    library = {
        "watched": [
            {
                "_id": "tt1234567",
                "_external_ids": {"imdb_id": "tt1234567", "tmdb_id": 4242},
                "type": "movie",
            }
        ],
        "loved": [],
        "liked": [],
        "removed": [],
        "added": [],
    }

    watched_imdb, watched_tmdb = asyncio.run(
        RecommendationFiltering.get_exclusion_sets(
            None,
            library,
            content_type="movie",
        )
    )

    assert watched_imdb == {"tt1234567"}
    assert watched_tmdb == {4242}


def test_filter_candidates_matches_external_id_shapes():
    candidates = [
        {"id": 1},
        {"id": "2"},
        {"id": 3, "_external_ids": {"imdb_id": "tt3000000"}},
        {"id": 4, "external_ids": {"tmdb_id": 4000}},
        {"id": 5},
    ]

    result = RecommendationFiltering.filter_candidates(
        candidates,
        watched_imdb={"tt3000000"},
        watched_tmdb={1, 2, 4000},
    )

    assert [item["id"] for item in result] == [5]
