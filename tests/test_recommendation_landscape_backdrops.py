import asyncio

from app.services.recommendation.metadata import RecommendationMetadata


class FakeTMDBService:
    def __init__(self) -> None:
        self.image_calls: list[int] = []

    async def get_movie_details(self, tmdb_id: int) -> dict:
        return {
            1: {
                "id": 1,
                "title": "Has Landscape Art",
                "backdrop_path": "/landscape.jpg",
                "poster_path": "/poster.jpg",
                "external_ids": {"imdb_id": "tt0000001"},
                "genres": [],
            },
            2: {
                "id": 2,
                "title": "Missing Landscape Art",
                "backdrop_path": None,
                "poster_path": "/poster2.jpg",
                "external_ids": {"imdb_id": "tt0000002"},
                "genres": [],
            },
        }[tmdb_id]

    async def get_tv_details(self, tmdb_id: int) -> dict:
        raise AssertionError(f"Unexpected TV details request for {tmdb_id}")

    async def get_images_for_title(
        self,
        media_type: str,
        tmdb_id: int,
        language: str,
    ) -> dict[str, str]:
        self.image_calls.append(tmdb_id)
        return {"logo": f"https://example.test/{tmdb_id}-logo.png"}


def test_fetch_batch_rejects_missing_landscape_before_image_enrichment():
    tmdb = FakeTMDBService()

    results = asyncio.run(
        RecommendationMetadata.fetch_batch(
            tmdb,
            [{"id": 1}, {"id": 2}],
            "movie",
        )
    )

    assert [item["id"] for item in results] == ["tt0000001"]
    assert results[0]["background"] == (
        "https://image.tmdb.org/t/p/original/landscape.jpg"
    )
    assert tmdb.image_calls == [1]


def test_fetch_batch_defensively_rejects_formatted_item_without_background():
    class MissingBackgroundFormatter(RecommendationMetadata):
        @classmethod
        async def format_for_stremio(
            cls,
            details,
            media_type,
            user_settings=None,
            logo_url=None,
        ):
            return {
                "id": "tt0000001",
                "type": "movie",
                "name": "Formatter Lost Background",
            }

    tmdb = FakeTMDBService()

    results = asyncio.run(
        MissingBackgroundFormatter.fetch_batch(
            tmdb,
            [{"id": 1}],
            "movie",
        )
    )

    assert results == []
    assert tmdb.image_calls == [1]
