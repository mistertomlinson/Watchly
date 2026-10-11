import asyncio
from datetime import date
from types import SimpleNamespace

from app.services.recommendation.availability import (
    _has_home_release,
    filter_home_released_movies,
)
from app.services.recommendation.top_picks import (
    TopPicksService,
    UNIFIED_TOP_PICKS_V2,
)
from app.services.recommendation.top_picks_v2 import TopPicksService as UnifiedTopPicksService


TODAY = date(2026, 10, 11)


class FakeClient:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    async def get(self, path):
        self.calls.append(path)
        tmdb_id = int(path.split("/")[2])
        return self.payloads.get(tmdb_id, {"results": []})


class FakeTmdbService:
    def __init__(self, payloads):
        self.client = FakeClient(payloads)


def release_payload(region, entries):
    return {
        "results": [
            {
                "iso_3166_1": region,
                "release_dates": entries,
            }
        ]
    }


def release_entry(release_type, release_date):
    return {
        "type": release_type,
        "release_date": f"{release_date}T00:00:00.000Z",
    }


def test_production_entry_point_is_unified_v2():
    assert UNIFIED_TOP_PICKS_V2 is True
    assert issubclass(TopPicksService, UnifiedTopPicksService)


def test_home_release_requires_digital_or_physical_not_theatrical():
    payload = release_payload("US", [release_entry(3, "2026-10-03")])

    assert not _has_home_release(
        payload,
        today=TODAY,
        preferred_region="US",
    )


def test_home_release_accepts_digital_or_physical_on_or_before_today():
    payload = release_payload(
        "US",
        [
            release_entry(3, "2026-09-20"),
            release_entry(4, "2026-10-10"),
            release_entry(5, "2026-10-15"),
        ],
    )

    assert _has_home_release(
        payload,
        today=TODAY,
        preferred_region="US",
    )


def test_region_bucket_wins_over_foreign_home_release_when_present():
    payload = {
        "results": [
            {
                "iso_3166_1": "US",
                "release_dates": [release_entry(3, "2026-10-03")],
            },
            {
                "iso_3166_1": "GB",
                "release_dates": [release_entry(4, "2026-10-05")],
            },
        ]
    }

    assert not _has_home_release(
        payload,
        today=TODAY,
        preferred_region="US",
    )


def test_recent_theatrical_only_movie_is_filtered():
    service = FakeTmdbService(
        {
            101: release_payload(
                "US",
                [release_entry(3, "2026-10-03")],
            )
        }
    )
    items = [
        {
            "id": "tt0000101",
            "_tmdb_id": 101,
            "name": "Theatrical Only",
            "released": "2026-10-03",
        }
    ]

    result = asyncio.run(
        filter_home_released_movies(
            service,
            items,
            SimpleNamespace(language="en-US"),
            today=TODAY,
        )
    )

    assert result == []
    assert service.client.calls == ["/movie/101/release_dates"]


def test_recent_movie_with_digital_release_is_kept():
    service = FakeTmdbService(
        {
            102: release_payload(
                "US",
                [release_entry(4, "2026-10-10")],
            )
        }
    )
    item = {
        "id": "tt0000102",
        "_tmdb_id": 102,
        "name": "Home Now",
        "released": "2026-09-25",
    }

    result = asyncio.run(
        filter_home_released_movies(
            service,
            [item],
            SimpleNamespace(language="en-US"),
            today=TODAY,
        )
    )

    assert result == [item]


def test_future_digital_release_is_not_yet_kept():
    service = FakeTmdbService(
        {
            103: release_payload(
                "US",
                [release_entry(4, "2026-11-01")],
            )
        }
    )
    items = [
        {
            "id": "tt0000103",
            "_tmdb_id": 103,
            "name": "Digital Next Month",
            "released": "2026-10-01",
        }
    ]

    result = asyncio.run(
        filter_home_released_movies(
            service,
            items,
            SimpleNamespace(language="en-US"),
            today=TODAY,
        )
    )

    assert result == []


def test_old_movie_does_not_depend_on_sparse_historical_home_release_metadata():
    service = FakeTmdbService({})
    item = {
        "id": "tt0000104",
        "_tmdb_id": 104,
        "name": "Old Catalog Movie",
        "released": "2020-01-01",
    }

    result = asyncio.run(
        filter_home_released_movies(
            service,
            [item],
            SimpleNamespace(language="en-US"),
            today=TODAY,
        )
    )

    assert result == [item]
    assert service.client.calls == []
