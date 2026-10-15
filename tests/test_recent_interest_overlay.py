import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.models.taste_profile import TasteProfile
from app.services.profile.integration import (
    ProfileIntegration,
    RecentInterestContext,
    RecentInterestOverlay,
)
from app.services.recommendation.catalog_service import CatalogService
from app.services.recommendation.top_picks_v2 import TopPicksService
from app.services.row_generator import (
    AXIS_GENRE,
    AXIS_KEYWORD,
    AxisRole,
    RowAxis,
    RowDefinition,
    RowGeneratorService,
    build_row_id,
)


class FakeSampler:
    @staticmethod
    def _is_taste_eligible(raw):
        return True


class FakeScoringService:
    @staticmethod
    def process_item(raw):
        watched_at = raw["state"]["lastWatched"]

        if isinstance(watched_at, str):
            watched_at = datetime.fromisoformat(
                watched_at.replace(
                    "Z",
                    "+00:00",
                )
            )

        return SimpleNamespace(
            item=SimpleNamespace(
                id=raw["_id"],
                type=raw["type"],
                name=raw["name"],
                state=SimpleNamespace(
                    lastWatched=watched_at
                ),
            )
        )


class FakeBuilder:
    def __init__(self, feature_map):
        self.feature_map = feature_map

    async def _process_item(
        self,
        scored,
        content_type,
    ):
        return (
            self.feature_map[
                scored.item.id
            ],
            0.5,
            False,
        )

    @staticmethod
    def _accumulate_features(
        profile,
        features,
        evidence_weight,
        is_loved,
        frequencies,
    ):
        for genre_id in (
            features.get(
                "genres",
                [],
            )
            or []
        ):
            profile.genre_scores[
                int(genre_id)
            ] = (
                profile.genre_scores.get(
                    int(genre_id),
                    0.0,
                )
                + evidence_weight
            )

        for keyword_id in (
            features.get(
                "keywords",
                [],
            )
            or []
        ):
            profile.keyword_scores[
                int(keyword_id)
            ] = (
                profile.keyword_scores.get(
                    int(keyword_id),
                    0.0,
                )
                + evidence_weight
            )

    @staticmethod
    def _apply_caps(profile):
        return None


def fake_integration(feature_map):
    integration = object.__new__(
        ProfileIntegration
    )

    integration.sampler = FakeSampler()
    integration.scoring_service = (
        FakeScoringService()
    )
    integration.builder = FakeBuilder(
        feature_map
    )

    return integration


def watched_item(
    media_id,
    name,
    watched_at,
    content_type="movie",
):
    return {
        "_id": media_id,
        "type": content_type,
        "name": name,
        "temp": False,
        "removed": False,
        "state": {
            "lastWatched":
                watched_at.isoformat(),
        },
    }


def test_recency_multiplier_anchor_points():
    assert (
        ProfileIntegration
        .recent_interest_multiplier(
            0
        )
        == pytest.approx(9.0)
    )

    assert (
        ProfileIntegration
        .recent_interest_multiplier(
            14
        )
        == pytest.approx(5.0)
    )

    assert (
        ProfileIntegration
        .recent_interest_multiplier(
            28
        )
        == pytest.approx(3.0)
    )


@pytest.mark.asyncio
async def test_movie_overlay_uses_only_recent_ordinary_watches():
    now = datetime(
        2026,
        10,
        14,
        12,
        0,
        tzinfo=timezone.utc,
    )

    profile = TasteProfile(
        content_type="movie",
        genre_scores={
            878: 10.0,
        },
        keyword_scores={
            10: 1.0,
        },
        processed_items={
            "tt1",
            "tt2",
            "tt3",
            "tt4",
        },
    )

    feature_map = {
        "tt1": {
            "genres": [878],
            "keywords": [
                10,
                20,
            ],
        },
        "tt2": {
            "genres": [28],
            "keywords": [
                10,
                20,
            ],
        },
        "tt3": {
            "genres": [28],
            "keywords": [30],
        },
        "tt4": {
            "genres": [18],
            "keywords": [40],
        },
    }

    library = {
        "watched": [
            watched_item(
                "tt1",
                "Recent One",
                now,
            ),
            watched_item(
                "tt2",
                "Recent Two",
                now
                - timedelta(days=1),
            ),
            watched_item(
                "tt3",
                "Explicit Like",
                now,
            ),
            watched_item(
                "tt4",
                "Too Old",
                now
                - timedelta(days=61),
            ),
        ],
        "liked": [
            watched_item(
                "tt3",
                "Explicit Like",
                now,
            ),
        ],
        "loved": [],
        "added": [],
        "disliked": [],
    }

    integration = fake_integration(
        feature_map
    )

    overlay = await (
        integration
        .build_recent_interest_overlay(
            profile,
            library,
            "movie",
            now=now,
        )
    )

    assert overlay.item_count == 2

    assert (
        overlay.context
        .repeated_keyword_ids
        == frozenset(
            {
                10,
                20,
            }
        )
    )

    assert (
        overlay.context
        .related_keyword_pairs
        == frozenset(
            {
                (10, 20),
            }
        )
    )

    assert (
        overlay.profile.keyword_scores[
            10
        ]
        > profile.keyword_scores[10]
    )

    assert 30 not in (
        overlay.profile.keyword_scores
    )

    assert 40 not in (
        overlay.profile.keyword_scores
    )

    # Durable profile remains unchanged.
    assert (
        profile.keyword_scores
        == {
            10: 1.0,
        }
    )


@pytest.mark.asyncio
async def test_series_overlay_boosts_one_series_identity_once():
    now = datetime(
        2026,
        10,
        14,
        12,
        0,
        tzinfo=timezone.utc,
    )

    profile = TasteProfile(
        content_type="series",
        genre_scores={
            18: 4.0,
        },
        keyword_scores={
            900: 1.0,
        },
        processed_items={
            "ttseries1",
        },
    )

    integration = fake_integration(
        {
            "ttseries1": {
                "genres": [18],
                "keywords": [
                    900,
                    901,
                ],
            },
        }
    )

    library = {
        "watched": [
            watched_item(
                "ttseries1",
                "Recent Series",
                now,
                content_type="series",
            ),
            # Defensive duplicate:
            # multiple history records for one show identity
            # must still create only one recency signal.
            watched_item(
                "ttseries1",
                "Recent Series",
                now - timedelta(hours=2),
                content_type="series",
            ),
        ],
        "liked": [],
        "loved": [],
        "added": [],
        "disliked": [],
    }

    overlay = await (
        integration
        .build_recent_interest_overlay(
            profile,
            library,
            "series",
            now=now,
        )
    )

    assert overlay.item_count == 1

    assert overlay.profile is not profile

    assert (
        overlay.profile.keyword_scores[900]
        > profile.keyword_scores[900]
    )

    assert (
        overlay.profile.keyword_scores[901]
        > 0
    )

    # Durable profile remains unchanged.
    assert (
        profile.keyword_scores
        == {
            900: 1.0,
        }
    )


@pytest.mark.asyncio
async def test_top_picks_discover_uses_candidate_profile():
    scoring_profile = TasteProfile(
        content_type="movie",
        genre_scores={
            28: 100.0,
        },
    )

    durable_profile = TasteProfile(
        content_type="movie",
        genre_scores={
            878: 100.0,
        },
    )

    service = TopPicksService(
        tmdb_service=SimpleNamespace(),
        user_settings=None,
    )

    service._fetch_recommendations_from_top_items = AsyncMock(
        return_value=[]
    )

    service._fetch_discover_with_profile = AsyncMock(
        return_value=[]
    )

    result = await service.get_top_picks(
        profile=scoring_profile,
        candidate_profile=durable_profile,
        content_type="movie",
        library_items={},
        watched_tmdb=set(),
        watched_imdb=set(),
        limit=20,
    )

    assert result == []

    service._fetch_discover_with_profile.assert_awaited_once()

    assert (
        service
        ._fetch_discover_with_profile
        .await_args
        .args[0]
        is durable_profile
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        "movie",
        "series",
    ],
)
async def test_catalog_router_separates_candidate_and_scoring_profiles(
    content_type,
):
    durable_profile = TasteProfile(
        content_type=content_type,
        genre_scores={
            878: 5.0,
        },
    )

    scoring_profile = (
        durable_profile.model_copy(
            deep=True
        )
    )

    scoring_profile.genre_scores[
        28
    ] = 20.0

    overlay = RecentInterestOverlay(
        profile=scoring_profile,
        context=RecentInterestContext(),
        item_count=2,
    )

    integration = SimpleNamespace(
        build_recent_interest_overlay=(
            AsyncMock(
                return_value=overlay
            )
        )
    )

    top_picks = SimpleNamespace(
        get_top_picks=AsyncMock(
            return_value=[
                {
                    "id": "tt0000001",
                    "type": content_type,
                    "name": "Movie",
                }
            ]
        )
    )

    service = CatalogService()

    result = await service._get_recommendations(
        catalog_id="watchly.rec",
        content_type=content_type,
        services={
            "top_picks": top_picks,
            "integration": integration,
        },
        profile=durable_profile,
        watched_tmdb=set(),
        watched_imdb=set(),
        whitelist=set(),
        library_items={
            "watched": [],
        },
        limit=20,
        user_settings=SimpleNamespace(
            openrouter_api_key=None
        ),
    )

    assert result

    kwargs = (
        top_picks
        .get_top_picks
        .await_args
        .kwargs
    )

    assert (
        kwargs["profile"]
        is scoring_profile
    )

    assert (
        kwargs["candidate_profile"]
        is durable_profile
    )


def test_theme_signal_signature_tracks_meaningful_recent_context():
    profile = TasteProfile(
        content_type="movie",
        genre_scores={
            878: 10.0,
            28: 9.0,
        },
        keyword_scores={
            4565: 10.0,
            9715: 9.0,
            180547: 8.0,
        },
    )

    no_recent = (
        RowGeneratorService
        .profile_signal_signature(
            profile,
            RecentInterestContext(),
        )
    )

    recent = (
        RowGeneratorService
        .profile_signal_signature(
            profile,
            RecentInterestContext(
                repeated_keyword_ids=(
                    frozenset(
                        {
                            9715,
                            180547,
                        }
                    )
                ),
                related_keyword_pairs=(
                    frozenset(
                        {
                            (
                                9715,
                                180547,
                            ),
                        }
                    )
                ),
            ),
        )
    )

    assert no_recent != recent

    same_identity_different_scores = (
        profile.model_copy(
            deep=True
        )
    )

    same_identity_different_scores.genre_scores[
        878
    ] += 0.01

    same_identity_different_scores.keyword_scores[
        4565
    ] += 0.01

    assert (
        RowGeneratorService
        .profile_signal_signature(
            same_identity_different_scores,
            RecentInterestContext(),
        )
        == no_recent
    )


@pytest.mark.asyncio
async def test_theme_cache_rejects_incompatible_signal_signature():
    row_generator = (
        RowGeneratorService(
            tmdb_service=SimpleNamespace()
        )
    )

    payload = json.dumps(
        {
            "generated_at":
                time.time(),
            "profile_signature":
                "old-signal",
            "rows": [],
        }
    )

    with patch(
        "app.services.redis_service.redis_service.get",
        new=AsyncMock(
            return_value=payload
        ),
    ):
        result = await (
            row_generator
            ._get_cached_llm_rows(
                "token",
                "movie",
                max_age=None,
                expected_signature="new-signal",
            )
        )

    assert result is None


def theme_row(
    title,
    genre_id,
    keyword_id=None,
):
    axes = [
        RowAxis(
            name=AXIS_GENRE,
            value=genre_id,
            role=AxisRole.ANCHOR,
        )
    ]

    if keyword_id is not None:
        axes.append(
            RowAxis(
                name=AXIS_KEYWORD,
                value=keyword_id,
                role=AxisRole.FLAVOR,
            )
        )

    return RowDefinition(
        title=title,
        id=build_row_id(axes),
        axes=axes,
    )


@pytest.mark.asyncio
async def test_theme_v2_allows_only_one_related_recent_interest_shelf():
    row_generator = (
        RowGeneratorService(
            tmdb_service=SimpleNamespace()
        )
    )

    rows = [
        theme_row(
            "Recent Theme A",
            878,
            100,
        ),
        theme_row(
            "Recent Theme B",
            28,
            200,
        ),
        theme_row(
            "Different Theme",
            18,
            300,
        ),
        theme_row(
            "Fourth Theme",
            35,
        ),
        theme_row(
            "Fifth Theme",
            80,
        ),
        theme_row(
            "Sixth Theme",
            53,
        ),
    ]

    names = {
        100: "recent-a",
        200: "recent-b",
        300: "different",
    }

    features = SimpleNamespace(
        keywords=[
            (100, 10.0),
            (200, 9.0),
            (300, 8.0),
        ],
        get_keyword_name=lambda keyword_id:
            names.get(
                int(keyword_id),
                str(keyword_id),
            ),
    )

    async def healthy_repair(
        row,
        content_type,
        watched_tmdb,
    ):
        return {
            "status": "healthy",
            "row": row,
            "probe": {
                "unseen_count": 50,
            },
            "changed": False,
            "repair_reason": None,
            "original_probe": {
                "unseen_count": 50,
            },
            "attempts": [],
        }

    row_generator._get_v2_normalized_watched_tmdb = (
        AsyncMock(
            return_value=set()
        )
    )

    row_generator._v2_repair_row_for_unseen = (
        AsyncMock(
            side_effect=healthy_repair
        )
    )

    result = await (
        row_generator
        ._select_v2_publishable_rows(
            rows,
            features,
            "movie",
            "token",
            history=[],
            recent_interest=(
                RecentInterestContext(
                    repeated_keyword_ids=(
                        frozenset(
                            {
                                100,
                                200,
                            }
                        )
                    ),
                    related_keyword_pairs=(
                        frozenset(
                            {
                                (
                                    100,
                                    200,
                                ),
                            }
                        )
                    ),
                )
            ),
        )
    )

    assert result is not None
    assert len(result) == 5

    result_ids = {
        row.id
        for row in result
    }

    # First related current-interest shelf survives.
    assert rows[0].id in result_ids

    # A second shelf from the same repeatedly co-occurring
    # recent-interest family is rejected.
    assert rows[1].id not in result_ids

    # Unrelated inventory can still fill all five slots.
    assert rows[2].id in result_ids
    assert rows[5].id in result_ids
