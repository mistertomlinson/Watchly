import pytest

from app.services.simkl_provider import SimklLibraryProvider


class FakeSimklClient:
    async def get_all_items(self, media_type):
        if media_type == "movie":
            return [
                {
                    "movie": {"title": "Loved Movie", "year": 2024, "ids": {"imdb": "tt1000001"}},
                    "status": "completed",
                    "user_rating": 10,
                    "last_watched_at": "2026-08-01T10:00:00Z",
                },
                {
                    "movie": {"title": "Neutral Movie", "year": 2023, "ids": {"imdb": "tt1000002"}},
                    "status": "completed",
                    "user_rating": 5,
                    "last_watched_at": "2026-07-01T10:00:00Z",
                },
            ]
        if media_type == "tv":
            return {
                "shows": [
                    {
                        "show": {"title": "Liked Show", "year": 2022, "ids": {"tmdb": 101}},
                        "status": "watching",
                        "rating": 8,
                        "watched_episodes_count": 3,
                    },
                    {
                        "show": {"title": "Disliked Show", "year": 2021, "ids": {"tmdb": 102}},
                        "status": "dropped",
                        "user_rating": 1,
                        "watched_episodes_count": 1,
                    },
                ]
            }
        return {"completed": []}


@pytest.mark.asyncio
async def test_simkl_provider_applies_watchly_rating_policy():
    library = await SimklLibraryProvider(FakeSimklClient()).get_library_items()

    assert {item["name"] for item in library["loved"]} == {"Loved Movie"}
    assert {item["name"] for item in library["liked"]} == {"Liked Show"}
    assert {item["name"] for item in library["disliked"]} == {"Disliked Show"}

    watched_names = {item["name"] for item in library["watched"]}
    assert watched_names == {"Loved Movie", "Neutral Movie", "Liked Show", "Disliked Show"}
    assert "Neutral Movie" not in {item["name"] for item in library["loved"] + library["liked"]}


def test_extract_entries_accepts_documented_and_wrapped_shapes():
    provider = SimklLibraryProvider(FakeSimklClient())
    assert provider._extract_entries([{"id": 1}], "movies") == [{"id": 1}]
    assert provider._extract_entries({"movies": [{"id": 2}]}, "movies") == [{"id": 2}]
    assert provider._extract_entries({"completed": [{"id": 3}]}, "movies") == [{"id": 3}]
