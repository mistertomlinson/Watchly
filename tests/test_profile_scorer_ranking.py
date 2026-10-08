import pytest

from app.services.profile.scorer import ProfileScorer


class StubProfile:
    def __init__(self, normalized):
        self._normalized = normalized

    def normalize_for_ranking(self):
        return self._normalized


def _profile(**overrides):
    normalized = {
        "genres": {},
        "keywords": {},
        "cast": {},
        "directors": {},
        "eras": {},
    }
    normalized.update(overrides)
    return StubProfile(normalized)


def test_profile_score_is_normalized_when_all_signals_are_maxed():
    profile = _profile(
        genres={878: 1.0},
        keywords={42: 1.0},
        cast={10: 1.0},
        directors={20: 1.0},
        eras={"2010s": 1.0},
    )
    item = {
        "genre_ids": [878],
        "keyword_ids": [42],
        "credits": {
            "cast": [{"id": 10}],
            "crew": [{"id": 20, "job": "Director"}],
        },
        "release_date": "2015-01-01",
    }

    assert ProfileScorer.score_item(item, profile) == pytest.approx(1.0)


def test_genre_ranking_uses_position_weights():
    profile = _profile(genres={878: 1.0, 16: 0.0, 35: 0.0})
    item = {"genre_ids": [878, 16, 35]}

    expected_genre = 1.0 / (1.0 + 0.8 + 0.5)
    assert ProfileScorer.score_item(item, profile) == pytest.approx(expected_genre * 0.40)


def test_technical_credit_keywords_do_not_influence_ranking():
    profile = _profile(keywords={179430: 1.0, 179431: 1.0})
    item = {
        "keywords": {
            "keywords": [
                {"id": 179430, "name": "aftercreditsstinger"},
                {"id": 179431, "name": "duringcreditsstinger"},
            ]
        }
    }

    assert ProfileScorer.score_item(item, profile) == pytest.approx(0.0)


def test_keyword_ranking_uses_strongest_meaningful_matches_only():
    profile = _profile(
        keywords={
            1: 1.0,
            2: 0.8,
            3: 0.6,
            4: 0.4,
            5: 0.2,
            6: 0.1,
            179431: 1.0,
        }
    )
    item = {
        "keyword_ids": [1, 2, 3, 4, 5, 6, 179431],
    }

    strongest_five_average = (1.0 + 0.8 + 0.6 + 0.4 + 0.2) / 5
    assert ProfileScorer.score_item(item, profile) == pytest.approx(strongest_five_average * 0.40)


def test_director_and_era_are_secondary_signals():
    profile = _profile(directors={20: 1.0}, eras={"2010s": 1.0})
    item = {
        "credits": {"crew": [{"id": 20, "job": "Director"}]},
        "release_date": "2014-01-01",
    }

    assert ProfileScorer.score_item(item, profile) == pytest.approx(0.10)
