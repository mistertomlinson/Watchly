from math import isinf

from app.services.profile.constants import CAP_GENRE
from app.services.profile.scorer import ProfileScorer
from app.services.recommendation.top_picks_v2 import TopPicksService


def test_candidate_merge_normalizes_tmdb_int_and_string_identity():
    pool = {}

    assert TopPicksService._merge_candidate(
        pool,
        {"id": "2048", "title": "I, Robot", "vote_count": 0},
        "simkl",
    )
    assert not TopPicksService._merge_candidate(
        pool,
        {"id": 2048, "title": "I, Robot", "vote_count": 1000},
        "gemini",
    )

    assert list(pool) == [2048]
    assert pool[2048]["id"] == 2048
    assert pool[2048]["vote_count"] == 1000
    assert pool[2048]["_watchly_sources"] == ["gemini", "simkl"]


def test_profile_scorer_directors_ignore_producers_and_deduplicate():
    item = {
        "credits": {
            "crew": [
                {"id": 1, "job": "Director"},
                {"id": 1, "job": "Director"},
                {"id": 2, "job": "Creator"},
                {"id": 3, "job": "Producer"},
            ]
        }
    }

    assert ProfileScorer._extract_director_ids(item) == [1, 2]


def test_genre_profile_evidence_is_no_longer_hard_capped_at_50():
    assert isinf(CAP_GENRE)
