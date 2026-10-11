import asyncio
from types import SimpleNamespace

from app.services.recommendation.metadata import RecommendationMetadata
from app.services.recommendation.top_picks import (
    TopPicksService,
    UNIFIED_TOP_PICKS_V2,
)
from app.services.recommendation.top_picks_v2 import TopPicksService as UnifiedTopPicksService


class FakeClient:
    def __init__(self, release_payloads):
        self.release_payloads = release_payloads
        self.calls = []

    async def get(self, path):
        self.calls.append(path)
        tmdb_id = int(path.split("/")[2])
        return self.release_payloads.get(tmdb_id, {"results": []})


class FakeTmdbService:
    def __init__(self, details, release_payloads):
        self.details = details
        self.client = FakeClient(release_payloads)

    async def get_movie_details(self, tmdb_id):
        return self.details.get(tmdb_id)

    async def get_images_for_title(self, media_type, tmdb_id, language="en-US"):
        return {}


def release_entry(release_type, release_date):
    return {
        "type": release_type,
        "release_date": f"{release_date}T00:00:00.000Z",
    }


def release_payload(entries):
    return {
        "results": [
            {
                "iso_3166_1": "US",
                "release_dates": entries,
            }
        ]
    }


def movie_details(tmdb_id, title, release_date):
    return {
        "id": tmdb_id,
        "title": title,
        "release_date": release_date,
        "external_ids": {"imdb_id": f"tt{tmdb_id:07d}"},
        "poster_path": "/poster.jpg",
        "backdrop_path": "/backdrop.jpg",
        "overview": "Test movie",
        "runtime": 110,
        "vote_average": 7.4,
        "vote_count": 1000,
        "popularity": 40.0,
        "original_language": "en",
        "genres": [{"id": 18, "name": "Drama"}],
        "credits": {"cast": []},
        "keywords": {"keywords": []},
    }


def test_top_picks_production_entry_point_is_unified_v2():
    assert UNIFIED_TOP_PICKS_V2 is True
    assert TopPicksService is UnifiedTopPicksService


def test_shared_metadata_boundary_filters_theatrical_only_movie():
    details = {
        201: movie_details(201, "Theatrical Only", "2026-10-03"),
        202: movie_details(202, "Home Now", "2026-09-25"),
    }
    releases = {
        201: release_payload([release_entry(3, "2026-10-03")]),
        202: release_payload(
            [
                release_entry(3, "2026-09-20"),
                release_entry(4, "2026-10-10"),
            ]
        ),
    }
    service = FakeTmdbService(details, releases)
    settings = SimpleNamespace(language="en-US", poster_rating=None)

    result = asyncio.run(
        RecommendationMetadata.fetch_batch(
            service,
            [{"id": 201}, {"id": 202}],
            "movie",
            user_settings=settings,
        )
    )

    assert [item["name"] for item in result] == ["Home Now"]
    assert "/movie/201/release_dates" in service.client.calls
    assert "/movie/202/release_dates" in service.client.calls
