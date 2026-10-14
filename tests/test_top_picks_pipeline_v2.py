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

def test_enrichment_selection_balances_sources_and_spreads_across_each_pool():
    pool = {}
    for index in range(100):
        TopPicksService._merge_candidate(
            pool,
            {"id": index + 1, "title": f"Simkl {index}"},
            "simkl",
        )
    for index in range(100):
        TopPicksService._merge_candidate(
            pool,
            {"id": index + 1001, "title": f"Gemini {index}"},
            "gemini",
        )

    selected = TopPicksService._select_for_enrichment(pool, 80)

    assert len(selected) == 80
    simkl = [item for item in selected if "simkl" in item["_watchly_sources"]]
    gemini = [item for item in selected if "gemini" in item["_watchly_sources"]]
    assert len(simkl) == 40
    assert len(gemini) == 40
    assert max(item["id"] for item in simkl) > 90
    assert max(item["id"] for item in gemini) > 1090

def test_strength_qualification_expands_beyond_display_target_when_matches_are_strong():
    scored = [(1.00, {}), (0.90, {}), (0.81, {}), (0.80, {}), (0.79, {})]
    assert TopPicksService._strength_qualified_count(scored, 2) == 4

def test_strength_qualification_uses_display_target_as_minimum_not_fixed_pool():
    scored = [(1.00, {}), (0.81, {}), (0.79, {}), (0.60, {}), (0.40, {})]
    assert TopPicksService._strength_qualified_count(scored, 4) == 4

def test_strength_qualification_has_no_maximum_pool_size():
    scored = [(1.00 - index * 0.001, {}) for index in range(50)]
    assert TopPicksService._strength_qualified_count(scored, 20) == 50

def test_post_dedupe_reserve_uses_next_best_scored_when_exactly_20_are_displayed():
    items = [
        {"id": f"tt{index:07d}", "_tmdb_id": index}
        for index in range(1, 76)
    ]
    displayed = items[:20]
    scored = [
        (100.0 - index, item)
        for index, item in enumerate(items)
    ]
    reserve = []

    displayed_before = [item["id"] for item in displayed]

    TopPicksService._fill_post_dedupe_reserve(
        reserve,
        displayed,
        scored,
        target_size=10,
    )

    assert [item["id"] for item in displayed] == displayed_before
    assert [item["id"] for item in reserve] == [
        item["id"] for item in items[20:30]
    ]


def test_post_dedupe_reserve_preserves_existing_qualified_extras_first():
    items = [
        {"id": f"tt{index:07d}", "_tmdb_id": index}
        for index in range(1, 40)
    ]
    displayed = items[:20]
    reserve = items[20:24].copy()
    scored = [
        (100.0 - index, item)
        for index, item in enumerate(items)
    ]

    TopPicksService._fill_post_dedupe_reserve(
        reserve,
        displayed,
        scored,
        target_size=10,
    )

    assert [item["id"] for item in reserve[:4]] == [
        item["id"] for item in items[20:24]
    ]
    assert [item["id"] for item in reserve] == [
        item["id"] for item in items[20:30]
    ]
