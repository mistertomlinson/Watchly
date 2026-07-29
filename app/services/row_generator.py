"""
Dynamic Row Generator Service.

Generates 3 personalized catalog rows using a tiered sampling system:
- Row 1 (The Core): User's strongest preferences (Gold tier: Top 1-3)
- Row 2 (The Blend): Mixed preferences with higher complexity (Gold+Silver: Top 1-8)
- Row 3 (The Rising Star): Emerging interests (Silver tier: Rank 4-10)
"""

import asyncio
import json
import random
from enum import Enum
from typing import Any, ClassVar

from loguru import logger
from pydantic import BaseModel, Field

from app.models.taste_profile import TasteProfile
from app.services.openrouter import gemini_service
from app.services.tmdb.countries import COUNTRY_ADJECTIVES
from app.services.tmdb.genre import movie_genres, series_genres
from app.services.tmdb.service import TMDBService, get_tmdb_service

GOLD_TIER_LIMIT = 3  # Top 1-3 items
SILVER_TIER_START = 3  # Rank 4+
SILVER_TIER_END = 10  # Up to Rank 10

# Available axes for row generation
AXIS_GENRE = "genre"
AXIS_KEYWORD = "keyword"
AXIS_COUNTRY = "country"
AXIS_RUNTIME = "runtime"
AXIS_CREATOR = "creator"


class AxisRole(str, Enum):
    ANCHOR = "anchor"  # strong signal, near-required
    FLAVOR = "flavor"  # boosts relevance, optional
    FALLBACK = "fallback"  # ranking only, never filtering


class RowAxis(BaseModel):
    name: str
    value: Any
    role: AxisRole
    weight: float = 1.0
    # Human-readable form of `value` (genre name, keyword name, country name).
    # Recorded at add time so titles can be composed from the axes themselves
    # rather than from a flat, order-dependent list of strings.
    display: str | None = None


def normalize_keyword(kw: str) -> str:
    """Normalize keyword for display."""
    return kw.strip().replace("-", " ").replace("_", " ").title()


def get_genre_name(genre_id: int, content_type: str) -> str:
    """Get genre name from ID."""
    genre_map = movie_genres if content_type == "movie" else series_genres
    return genre_map.get(genre_id, "Movies" if content_type == "movie" else "Series")


def get_country_adjective(country_code: str) -> str | None:
    """Get country adjective (e.g., 'US' -> 'American')."""
    adjectives = COUNTRY_ADJECTIVES.get(country_code, [])
    return random.choice(adjectives) if adjectives else None


def runtime_to_modifier(bucket: str) -> str | None:
    """Get display modifier for runtime bucket."""
    modifiers = {
        "short": "Short & Sweet",
        "medium": None,  # No modifier for medium
        "long": "Epic",
    }
    return modifiers.get(bucket)


def sample_from_tier(items: list[tuple[Any, float]], start: int, end: int, count: int = 1) -> list[tuple[Any, float]]:
    """Sample random items from a specific tier range."""
    tier_items = items[start:end]
    if not tier_items:
        return []
    return random.sample(tier_items, min(count, len(tier_items)))


def sample_from_gold(items: list[tuple[Any, float]], count: int = 1) -> list[tuple[Any, float]]:
    """Sample from Gold tier (Top 1-3)."""
    return sample_from_tier(items, 0, GOLD_TIER_LIMIT, count)


def sample_from_silver(items: list[tuple[Any, float]], count: int = 1) -> list[tuple[Any, float]]:
    """Sample from Silver tier (Rank 4-10)."""
    return sample_from_tier(items, SILVER_TIER_START, SILVER_TIER_END, count)


def sample_from_gold_silver(items: list[tuple[Any, float]], count: int = 1) -> list[tuple[Any, float]]:
    """Sample from combined Gold+Silver tier (Rank 1-10)."""
    return sample_from_tier(items, 0, SILVER_TIER_END, count)


def build_row_id(axes: list[RowAxis]) -> str:
    """Build a unique row ID from axes and their roles."""
    parts = ["watchly.theme"]

    role_map = {
        AxisRole.ANCHOR: "a",
        AxisRole.FLAVOR: "f",
        AxisRole.FALLBACK: "b",
    }

    # Sort axes for consistent IDs
    sorted_axes = sorted(axes, key=lambda x: (x.role, x.name, str(x.value)))

    for axis in sorted_axes:
        role_pfx = role_map.get(axis.role, "f")
        axis_pfx = {
            AXIS_GENRE: "g",
            AXIS_KEYWORD: "k",
            AXIS_COUNTRY: "ct",
            AXIS_RUNTIME: "r",
            AXIS_CREATOR: "cr",
        }.get(axis.name, "x")

        # Handle value formatting
        val = axis.value
        if isinstance(val, (list, tuple)):
            val = "-".join(str(v) for v in val)

        parts.append(f"{role_pfx}:{axis_pfx}{val}")

    return ".".join(parts)


class RowDefinition(BaseModel):
    """Defines a dynamic catalog row."""

    title: str
    id: str
    axes: list[RowAxis] = []
    explanation: str | None = None
    expansion_strategy: str | None = None

    @property
    def is_valid(self) -> bool:
        return len(self.axes) > 0


class LLMRowTheme(BaseModel):
    """Schema for structured LLM output - a single themed catalog row."""

    title: str = Field(
        description=(
            "2-5 word title describing the CONTENT. Never name the row's purpose "
            "(no core/mixed/rising/deep cut/mood/picks/favorites)."
        )
    )
    genres: list[int] = Field(description="List of valid TMDB genre IDs")
    keywords: list[str] = Field(default_factory=list, description="Specific TMDB keyword names")
    country: str | None = Field(default=None, description="ISO 3166-1 country code or null")


class RowComponents(BaseModel):
    """Internal structure for building a row."""

    axes: list[RowAxis] = []
    explanation: str | None = None

    # For title generation
    prompt_parts: list[str] = []
    fallback_parts: list[str] = []

    def build_prompt(self) -> str:
        """Build Gemini prompt from parts."""
        return " + ".join(self.prompt_parts)

    # TMDB keyword names read as sentence fragments and make poor titles verbatim.
    KEYWORD_DISPLAY_OVERRIDES: ClassVar[dict[str, str]] = {
        "based on novel or book": "Novel-Based",
        "based on true story": "True Story",
        "based on comic": "Comic-Based",
        "artificial intelligence (a.i.)": "AI",
        "true crime": "True Crime",
        "dystopia": "Dystopian",
        "murder investigation": "Murder Mystery",
        "dark comedy": "Dark",
        "woman director": "Women-Directed",
    }

    def build_fallback(self) -> str:
        """Compose a readable title from the row's axes.

        Previously this joined fallback_parts in whatever order they happened to be
        added, producing titles like "Based On Novel Or Book Mystery". Ordering by
        axis type -- country, then keyword, then genre as the closing noun -- gives
        "Novel-Based Mystery" instead.
        """
        by_axis: dict[str, list[str]] = {}
        for axis in self.axes:
            if axis.role not in (AxisRole.ANCHOR, AxisRole.FLAVOR):
                continue
            display = getattr(axis, "display", None) or self._display_for(axis)
            if not display:
                continue
            by_axis.setdefault(axis.name, []).append(display)

        countries = by_axis.get(AXIS_COUNTRY, [])[:1]
        keywords = by_axis.get(AXIS_KEYWORD, [])[:2]
        genres = by_axis.get(AXIS_GENRE, [])[:2]

        keywords = [self.KEYWORD_DISPLAY_OVERRIDES.get(k.strip().lower(), k) for k in keywords]

        # Drop a keyword that merely restates a genre (e.g. "Dystopian" + "Science Fiction")
        genre_words = {g.strip().lower() for g in genres}
        keywords = [k for k in keywords if k.strip().lower() not in genre_words]

        parts = countries + keywords + genres
        title = " ".join(p for p in parts if p).strip()

        # Fall back to the old behaviour if axis data was unavailable
        return title or " ".join(self.fallback_parts)

    def _display_for(self, axis) -> str | None:
        """Recover a display string for an axis from the recorded fallback parts."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for row building."""
        return {
            "axes": self.axes,
            "explanation": self.explanation,
        }


class ExtractedFeatures:
    """Container for all extracted profile features with keyword names resolved."""

    def __init__(
        self,
        genres: list[tuple[int, float]],
        keywords: list[tuple[int, float]],
        countries: list[tuple[str, float]],
        runtimes: list[tuple[str, float]],
        creators: list[tuple[int, float]],
        keyword_names: dict[int, str],
        content_type: str,
    ):
        self.genres = genres
        self.keywords = keywords
        self.countries = countries
        self.runtimes = runtimes
        self.creators = creators
        self.keyword_names = keyword_names
        self.content_type = content_type

    def get_keyword_name(self, keyword_id: int) -> str | None:
        return self.keyword_names.get(keyword_id)

    def get_genre_name(self, genre_id: int) -> str:
        return get_genre_name(genre_id, self.content_type)


class RowBuilder:
    """Builds a single row by sampling from axes with specific roles."""

    def __init__(self, features: ExtractedFeatures):
        self.features = features
        self.components = RowComponents()
        self.used_axes: set[str] = set()

    def add_axis(self, name: str, value: Any, role: AxisRole, weight: float = 1.0) -> "RowBuilder":
        """Add an axis with a specific role and weight."""
        axis = RowAxis(name=name, value=value, role=role, weight=weight)
        self.components.axes.append(axis)

        # Build prompt and fallback title parts
        display_val = self._get_display_value(name, value)
        axis.display = display_val
        if display_val:
            prefix = ""
            if role == AxisRole.ANCHOR:
                prefix = "Anchor: "
            elif role == AxisRole.FLAVOR:
                prefix = "Flavor: "

            self.components.prompt_parts.append(f"{prefix}{name.title()}: {display_val}")

            # For fallback title, we prioritize Anchor and Flavor
            if role in (AxisRole.ANCHOR, AxisRole.FLAVOR):
                if name == AXIS_COUNTRY:
                    self.components.fallback_parts.insert(0, display_val)
                else:
                    self.components.fallback_parts.append(display_val)

        self.used_axes.add(f"{name}:{value}")
        return self

    def _get_display_value(self, name: str, value: Any) -> str | None:
        """Get human-readable value for an axis."""
        if name == AXIS_GENRE:
            return self.features.get_genre_name(value)
        if name == AXIS_KEYWORD:
            return normalize_keyword(self.features.get_keyword_name(value) or "")
        if name == AXIS_COUNTRY:
            return get_country_adjective(value)
        if name == AXIS_RUNTIME:
            return runtime_to_modifier(value)
        return str(value)

    def build(self) -> RowComponents | None:
        """Build and return the row components if valid (has at least one anchor)."""
        has_anchor = any(a.role == AxisRole.ANCHOR for a in self.components.axes)
        if has_anchor:
            return self.components
        return None


# Minimum raw TMDB inventory a themed row must have before it is worth showing.
# Raw total_results is a generous upper bound: the user's year range, popularity
# floor and watched-history filtering all cut into it, so the effective pool is a
# good deal smaller than this number.
MIN_ROW_INVENTORY = 60

# Keywords too semantically empty to define a row. IDs verified against the TMDB
# keyword endpoint -- four of the original eight comments named the wrong keyword
# ("based on true story" was actually 818 = based on novel or book, "philosophical"
# was 4344 = musical, "suspense" was 663 = fortune teller, "based on novel" was
# 9717 = based on comic), which suppressed several perfectly good themes.
#
# Thin-but-meaningful keywords no longer need to be listed here: MIN_ROW_INVENTORY
# in _repair_thin_rows rejects anything without enough titles behind it.
GENERIC_KEYWORD_BLACKLIST = {
    239797,  # complex        - subjective, describes no subject matter
    197582,  # mysterious     - subjective, describes no subject matter
    2964,    # future         - too broad; overlaps the sci-fi genre
    11162,   # miniseries     - format descriptor, not content
}

# Previously blacklisted by mistake, now allowed:
#   818    based on novel or book (8257 titles)
#   9717   based on comic        (851 titles)
#   4344   musical               (3844 titles)
#   663    fortune teller        (79 titles - inventory check gates this one)

class RowGeneratorService:
    """Generates dynamic, personalized row definitions from a User Taste Profile."""

    def __init__(self, tmdb_service: TMDBService | None = None):
        self.tmdb_service = tmdb_service or get_tmdb_service()

    async def generate_rows(
        self,
        profile: TasteProfile,
        content_type: str = "movie",
        api_key: str | None = None,
        token: str | None = None,
        avoid_titles: set[str] | None = None,
    ) -> list[RowDefinition]:
        """
        Generate exactly 5 personalized catalog rows.
        If api_key is provided, uses LLM to generate creative themes.
        Otherwise uses tiered sampling system.

        Returns:
            List of RowDefinition
        """
        # 1. Extract all features from profile
        features = await self._extract_features(profile, content_type)

        # 2. Try LLM generation if key is present
        if api_key:
            try:
                llm_rows = await self._generate_rows_with_llm(
                    profile, features, content_type, api_key, avoid_titles
                )
                if llm_rows:
                    logger.info(f"Generated {len(llm_rows)} LLM-driven rows for {content_type}")
                    await self._cache_llm_rows(token, content_type, llm_rows)
                    return llm_rows
            except Exception as e:
                logger.warning(f"LLM row generation failed: {e}")

            # The LLM call failed (rate limit, quota, outage). Prefer the last set of
            # rows it produced over mechanically sampled ones -- otherwise a single
            # 429 gets cached in the manifest and degrades every row for hours.
            cached_rows = await self._get_cached_llm_rows(token, content_type)
            if cached_rows:
                logger.info(
                    f"Serving {len(cached_rows)} cached LLM rows for {content_type} "
                    "(live generation unavailable)"
                )
                return cached_rows

        # 3. Fallback to Tiered Sampling
        rows_data = []
        used_genres = set()
        used_keywords = set()

        # Row 1: The Core (Strongest matches)
        core_row = self._build_core_row(features, exclude_genres=used_genres, exclude_keywords=used_keywords)
        if core_row:
            rows_data.append(core_row)
            self._update_used_axes(core_row, used_genres, used_keywords)

        # Row 2: The Blend (Mixing themes)
        blend_row = self._build_blend_row(features, exclude_genres=used_genres, exclude_keywords=used_keywords)
        if blend_row:
            rows_data.append(blend_row)
            self._update_used_axes(blend_row, used_genres, used_keywords)

        # Row 3: The Rising Star (Exploration)
        rising_row = self._build_rising_star_row(features, exclude_genres=used_genres, exclude_keywords=used_keywords)
        if rising_row:
            rows_data.append(rising_row)

        # 4. Generate titles via server's default Gemini model (gemma)
        final_rows = await self._generate_titles(rows_data[:3])

        # Tiered sampling picks keywords purely by profile frequency, with no regard
        # for how many titles actually carry them. A niche keyword combined with a
        # genre can yield literally zero results (e.g. squatting + comedy on TV), so
        # the same inventory check applied to LLM rows is applied here.
        try:
            final_rows = await self._repair_thin_rows(final_rows, features, content_type)
        except Exception as e:
            logger.warning(f"[RowRepair] tiered-sampling repair failed: {e}")

        logger.info(f"Generated {len(final_rows)} dynamic rows (Tiered Sampling) for {content_type}")
        return final_rows

    @staticmethod
    def _llm_rows_key(token: str, content_type: str) -> str:
        return f"watchly:llm_rows:{token}:{content_type}"

    async def _cache_llm_rows(self, token: str | None, content_type: str, rows: list) -> None:
        """Persist the most recent successful LLM row set as a fallback."""
        if not token or not rows:
            return
        try:
            from app.services.redis_service import redis_service
            payload = json.dumps([r.model_dump(mode="json") for r in rows])
            await redis_service.set(self._llm_rows_key(token, content_type), payload, 604800)
        except Exception as e:
            logger.debug(f"[LLM Rows] failed to cache rows: {e}")

    async def _get_cached_llm_rows(self, token: str | None, content_type: str) -> list | None:
        """Return the last successful LLM row set, if one is stored."""
        if not token:
            return None
        try:
            from app.services.redis_service import redis_service
            raw = await redis_service.get(self._llm_rows_key(token, content_type))
            if not raw:
                return None
            return [RowDefinition(**d) for d in json.loads(raw)]
        except Exception as e:
            logger.debug(f"[LLM Rows] failed to read cached rows: {e}")
            return None

    def _update_used_axes(self, row: RowComponents, used_genres: set, used_keywords: set):
        """Track used genres and keywords to ensure row diversity."""
        for axis in row.axes:
            if axis.name == AXIS_GENRE:
                used_genres.add(axis.value)
            elif axis.name == AXIS_KEYWORD:
                used_keywords.add(axis.value)

    async def _extract_features(self, profile: TasteProfile, content_type: str) -> ExtractedFeatures:
        """Extract all features from profile and resolve keyword names."""
        # Get raw features
        genres = profile.get_top_genres(limit=5)
        keywords = profile.get_top_keywords(limit=10)
        countries = profile.get_top_countries(limit=2)
        runtimes = sorted(profile.runtime_bucket_scores.items(), key=lambda x: x[1], reverse=True)
        creators = profile.get_top_creators(limit=5)

        # Fetch keyword names in parallel
        keyword_ids = [k_id for k_id, _ in keywords]
        keyword_names_raw = await asyncio.gather(
            *[self._get_keyword_name(kid) for kid in keyword_ids],
            return_exceptions=True,
        )
        keyword_names = {
            kid: name for kid, name in zip(keyword_ids, keyword_names_raw) if name and not isinstance(name, Exception)
        }

        return ExtractedFeatures(
            genres=genres,
            keywords=keywords,
            countries=countries,
            runtimes=runtimes,
            creators=creators,
            keyword_names=keyword_names,
            content_type=content_type,
        )

    async def _get_keyword_name(self, keyword_id: int) -> str | None:
        """Fetch keyword name from TMDB."""
        try:
            data = await self.tmdb_service.get_keyword_details(keyword_id)
            return data.get("name")
        except Exception:
            return None

    def _build_core_row(
        self,
        features: ExtractedFeatures,
        exclude_genres: set[int] | None = None,
        exclude_keywords: set[int] | None = None,
    ) -> RowComponents | None:
        """
        Build 'The Core' row:
        Anchor: GENRE (Gold)
        Flavor: 1-2 KEYWORDS (Gold)
        Fallback: RUNTIME (Gold/Silver)
        """
        exclude_genres = exclude_genres or set()
        exclude_keywords = exclude_keywords or set()
        builder = RowBuilder(features)

        # 1. Anchor: Genre
        available_genres = [g for g in features.genres if g[0] not in exclude_genres]
        genres = sample_from_gold(available_genres, 1) if available_genres else sample_from_gold(features.genres, 1)
        if not genres:
            return None
        builder.add_axis(AXIS_GENRE, genres[0][0], AxisRole.ANCHOR, 1.0)

        # 2. Flavor: 1-2 Keywords
        available_keywords = [k for k in features.keywords if k[0] not in exclude_keywords]
        keywords = sample_from_gold(available_keywords, random.randint(1, 2)) if available_keywords else []
        for k_id, _ in keywords:
            builder.add_axis(AXIS_KEYWORD, k_id, AxisRole.FLAVOR, 0.7)

        # 3. Fallback: Runtime
        if features.runtimes:
            runtime = random.choice(features.runtimes[:2])
            builder.add_axis(AXIS_RUNTIME, runtime[0], AxisRole.FALLBACK, 0.3)

        row = builder.build()
        if row:
            row.explanation = "The Core: Based on your absolute favorite genres and recurring themes."
        return row

    def _build_blend_row(
        self,
        features: ExtractedFeatures,
        exclude_genres: set[int] | None = None,
        exclude_keywords: set[int] | None = None,
    ) -> RowComponents | None:
        """
        Build 'The Blend' row:
        Anchor: GENRE (Gold)
        Flavor: COUNTRY or secondary GENRE (Gold/Silver)
        """
        exclude_genres = exclude_genres or set()
        builder = RowBuilder(features)

        # 1. Anchor: Genre
        available_genres = [g for g in features.genres if g[0] not in exclude_genres]
        genres = sample_from_gold(available_genres, 1) if available_genres else sample_from_gold(features.genres, 1)
        if not genres:
            return None
        builder.add_axis(AXIS_GENRE, genres[0][0], AxisRole.ANCHOR, 1.0)

        # 2. Flavor: Country or Secondary Genre
        flavor_type = random.choice([AXIS_COUNTRY, AXIS_GENRE])

        if flavor_type == AXIS_COUNTRY and features.countries:
            country = sample_from_gold_silver(features.countries, 1)
            builder.add_axis(AXIS_COUNTRY, country[0][0], AxisRole.FLAVOR, 0.7)
        elif flavor_type == AXIS_GENRE:
            other_genres = [g for g in features.genres if g[0] != genres[0][0]]
            if other_genres:
                sec_genre = sample_from_gold_silver(other_genres, 1)
                builder.add_axis(AXIS_GENRE, sec_genre[0][0], AxisRole.FLAVOR, 0.7)

        row = builder.build()
        if row:
            row.explanation = "The Blend: Mixing your top genres with international flavor or secondary interests."
        return row

    def _build_rising_star_row(
        self,
        features: ExtractedFeatures,
        exclude_genres: set[int] | None = None,
        exclude_keywords: set[int] | None = None,
    ) -> RowComponents | None:
        """
        Build 'The Rising Star' row:
        Anchor: recent KEYWORD (Silver)
        Flavor: GENRE (Silver)
        Fallback: COUNTRY (Gold/Silver)
        """
        exclude_genres = exclude_genres or set()
        exclude_keywords = exclude_keywords or set()
        builder = RowBuilder(features)

        # 1. Anchor: Recent Keyword (Sampling from Silver to promote exploration)
        available_keywords = [k for k in features.keywords if k[0] not in exclude_keywords]
        keywords = sample_from_silver(available_keywords, 1) if available_keywords else []
        if keywords:
            builder.add_axis(AXIS_KEYWORD, keywords[0][0], AxisRole.ANCHOR, 1.0)

        # If we couldn't add an anchor, this row fails
        if not builder.components.axes:
            return None

        # 2. Flavor: Genre (Silver)
        available_genres = [g for g in features.genres if g[0] not in exclude_genres]
        genres = sample_from_silver(available_genres, 1) if available_genres else []
        if genres:
            builder.add_axis(AXIS_GENRE, genres[0][0], AxisRole.FLAVOR, 0.7)

        # 3. Fallback: Country
        if features.countries:
            country = sample_from_gold_silver(features.countries, 1)
            builder.add_axis(AXIS_COUNTRY, country[0][0], AxisRole.FALLBACK, 0.3)

        row = builder.build()
        if row:
            row.explanation = "The Rising Star: Exploring emerging interests and newer themes in your history."
        return row

    def _build_signature_rows(self, features: ExtractedFeatures) -> list[RowComponents]:
        """Generate dynamic signature recipes from user history."""
        signature_rows = []

        # 1. Top genre × dominant keyword
        if features.genres and features.keywords:
            builder = RowBuilder(features)
            builder.add_axis(AXIS_GENRE, features.genres[0][0], AxisRole.ANCHOR, 1.0)
            builder.add_axis(AXIS_KEYWORD, features.keywords[0][0], AxisRole.FLAVOR, 0.7)
            row = builder.build()
            if row:
                row.explanation = "Signature: Your #1 genre paired with your most frequent theme."
                signature_rows.append(row)

        # 2. Top genre × preferred runtime
        if features.genres and features.runtimes:
            builder = RowBuilder(features)
            builder.add_axis(AXIS_GENRE, features.genres[0][0], AxisRole.ANCHOR, 1.0)
            builder.add_axis(AXIS_RUNTIME, features.runtimes[0][0], AxisRole.FLAVOR, 0.7)
            row = builder.build()
            if row:
                row.explanation = "Signature: Favorite genre fit for your preferred watch duration."
                signature_rows.append(row)

        return signature_rows

    @staticmethod
    def _clean_title(title: str) -> str:
        """Strip wrapping quotes an LLM sometimes adds around a returned title.

        Models asked for "a single best title and nothing else" frequently answer
        with the title in quotes, which then ends up rendered literally in the
        catalog row name.
        """
        t = (title or "").strip()
        for _ in range(2):
            if len(t) >= 2 and t[0] in '"\'\u201c\u2018' and t[-1] in '"\'\u201d\u2019':
                t = t[1:-1].strip()
            else:
                break
        return t or title

    async def _generate_titles(self, rows_data: list[RowComponents]) -> list[RowDefinition]:
        """Generate titles for tiered sampling rows via server's default Gemini model."""
        if not rows_data:
            return []

        # Build prompts and fire Gemini requests (uses server key + default model)
        prompts = [row.build_prompt() for row in rows_data]
        gemini_tasks = [gemini_service.generate_content_async(p) for p in prompts]
        results = await asyncio.gather(*gemini_tasks, return_exceptions=True)

        final_rows = []
        for i, row in enumerate(rows_data):
            result = results[i]

            # Determine title
            if isinstance(result, Exception):
                logger.warning(f"Gemini failed for row {i}: {result}")
                title = row.build_fallback()
            elif result:
                title = self._clean_title(result)
            else:
                title = row.build_fallback()

            # Build the row ID
            row_id = build_row_id(row.axes)

            final_rows.append(
                RowDefinition(
                    title=title,
                    id=row_id,
                    **row.to_dict(),
                )
            )

        return final_rows

    async def _resolve_keyword_to_id(self, kw_name: str, profile_kw_map: dict[str, int]) -> int | None:
        """Resolve a keyword name to TMDB ID: profile first, then TMDB search (for discovery)."""
        kw_lower = str(kw_name).strip().lower()
        if not kw_lower:
            return None
        if kw_lower in profile_kw_map:
            return profile_kw_map[kw_lower]
        try:
            data = await self.tmdb_service.search_keywords(kw_lower)
            results = data.get("results") or []
            if results:
                first = results[0]
                kid = first.get("id") if isinstance(first, dict) else getattr(first, "id", None)
                if kid is not None:
                    return int(kid)
        except Exception:
            pass
        return None

    async def _row_inventory(self, axes: list, content_type: str) -> int:
        """Return TMDB's total_results for the discover query a row will run."""
        genres = [str(a.value) for a in axes if a.name == AXIS_GENRE]
        keywords = [str(a.value) for a in axes if a.name == AXIS_KEYWORD]
        countries = [str(a.value) for a in axes if a.name == AXIS_COUNTRY]

        params = {}
        if genres:
            params["with_genres"] = "|".join(genres)
        if keywords:
            params["with_keywords"] = "|".join(keywords)
        if countries:
            params["with_origin_country"] = countries[0]
        if not params:
            return 0

        try:
            res = await self.tmdb_service.get_discover(content_type, page=1, **params)
            return int(res.get("total_results", 0) or 0)
        except Exception as e:
            logger.debug(f"[RowRepair] inventory probe failed: {e}")
            # Fail open: never discard a row because of a transient TMDB error.
            return MIN_ROW_INVENTORY

    async def _repair_thin_rows(
        self,
        rows: list,
        features: "ExtractedFeatures",
        content_type: str,
    ) -> list:
        """Ensure every row can actually fill itself.

        A keyword the LLM picks may have almost no titles behind it (e.g. "squatting"
        has ~31 films on TMDB). Such a row previously padded itself with unrelated
        content. Here we instead swap the keyword for a viable one from the user's
        profile, or -- failing that -- drop the keyword axis and let the row stand on
        its genre/country. The row is never removed, so the catalog count is stable.
        """
        used_keywords = {a.value for r in rows for a in r.axes if a.name == AXIS_KEYWORD}

        for row in rows:
            kw_axes = [a for a in row.axes if a.name == AXIS_KEYWORD]
            if not kw_axes:
                continue

            inventory = await self._row_inventory(row.axes, content_type)
            if inventory >= MIN_ROW_INVENTORY:
                continue

            logger.info(
                f"[RowRepair] '{row.title}' has only {inventory} titles "
                f"(min {MIN_ROW_INVENTORY}); attempting repair"
            )

            non_kw = [a for a in row.axes if a.name != AXIS_KEYWORD]
            repaired = False

            # Step 1: try a different keyword from the profile.
            for kid, _score in features.keywords:
                if kid in used_keywords or kid in GENERIC_KEYWORD_BLACKLIST:
                    continue
                candidate = list(non_kw) + [
                    RowAxis(name=AXIS_KEYWORD, value=kid, role=AxisRole.FLAVOR)
                ]
                if await self._row_inventory(candidate, content_type) >= MIN_ROW_INVENTORY:
                    kw_name = normalize_keyword(features.get_keyword_name(kid) or "")
                    kw_name = RowComponents.KEYWORD_DISPLAY_OVERRIDES.get(
                        kw_name.strip().lower(), kw_name
                    )
                    genre_names = [
                        features.get_genre_name(a.value) for a in non_kw if a.name == AXIS_GENRE
                    ]
                    country_names = [
                        get_country_adjective(a.value) for a in non_kw if a.name == AXIS_COUNTRY
                    ]
                    # Drop a keyword that just restates a genre, and keep the genre
                    # last so the title reads as a noun phrase:
                    # [Country] [Keyword] [Genre] -> "British Novel-Based Drama"
                    if kw_name.strip().lower() in {g.strip().lower() for g in genre_names}:
                        kw_name = ""
                    row.axes = candidate
                    row.id = build_row_id(candidate)
                    row.title = (
                        " ".join(p for p in (country_names[:1] + [kw_name] + genre_names[:1]) if p)
                        or row.title
                    )
                    used_keywords.add(kid)
                    repaired = True
                    logger.info(f"[RowRepair] swapped in keyword {kid} -> '{row.title}'")
                    break

            # Step 2: keep the row, lose the keyword.
            if not repaired and non_kw:
                genre_names = [
                    features.get_genre_name(a.value) for a in non_kw if a.name == AXIS_GENRE
                ]
                country_names = [
                    get_country_adjective(a.value) for a in non_kw if a.name == AXIS_COUNTRY
                ]
                row.axes = non_kw
                row.id = build_row_id(non_kw)
                if genre_names:
                    row.title = " ".join(country_names[:1] + genre_names[:2])
                logger.info(f"[RowRepair] dropped keyword axis -> '{row.title}'")

        return rows

    async def _generate_rows_with_llm(
        self,
        profile: TasteProfile,
        features: ExtractedFeatures,
        content_type: str,
        api_key: str,
        avoid_titles: set[str] | None = None,
    ) -> list[RowDefinition] | None:
        """Generate rows from the user's interest summary; balance personalization with discovery."""
        try:
            summary = profile.interest_summary or "No summary available."

            current_genre_map = movie_genres if content_type == "movie" else series_genres
            valid_genre_list = ", ".join([f"{name} (ID: {gid})" for gid, name in current_genre_map.items()])

            profile_keywords = [name for k_id, _ in features.keywords[:12] if (name := features.get_keyword_name(k_id))]
            keyword_hint = (
                (
                    f"Themes they already like (you can use these): {', '.join(profile_keywords)}. "
                    if profile_keywords
                    else ""
                )
                + "You can also suggest new themes for discovery—especially for Rising Star—"
                "e.g. adjacent genres or topics they might not have tried yet. We will resolve keywords."
            )

            avoid_clause = ""
            if avoid_titles:
                avoid_clause = (
                    "\n\nALREADY USED — these exact titles are taken by another section of"
                    " this catalogue. Do not reuse them and do not produce near-identical"
                    f" variants: {', '.join(sorted(avoid_titles))}."
                )

            prompt = (
                "Using only the user's interest summary below, generate exactly 5 streaming collections for"
                f" {content_type}. Use genres (required), keywords, and country when relevant.\n\nInterest"
                f" Summary:\n{summary}\n\nGenerate 5 rows. The bracketed labels are internal planning labels only and must NEVER appear in a title:\n1. [strongest match] — What they will love"
                " most: strongest match to their taste (genres + keywords + country if relevant).\n2. [variety]"
                " — Blend of their tastes with more variety (genres + keywords + country if"
                " relevant).\n3. [discovery] — Discovery: suggest themes they might not have explored yet but"
                " would likely enjoy (adjacent to their taste, or natural next step). Use genres + keywords +"
                " country; openness to new content here.\n4. [lesser-known] — Lesser-known or cult titles matching"
                " their taste. Use 1-2 genres + 1 keyword max.\n5. [mood] — A specific mood or tone they"
                " would enjoy (e.g. mind-bending, atmospheric, tense). Use genres + 1 keyword max.\n\nRules:\n"
                "- Genres: use ONLY these TMDB Genre IDs:"
                f" {valid_genre_list}\n- Keywords: {keyword_hint}\n- Country: ISO 3166-1 alpha-2 code (e.g. US, KR, JP, GB) or null. NEVER use UK — use GB for Britain/England."
                " or null when relevant.\n- TITLE RULE (most important): the title must describe WHAT THE FILMS ARE, never the row's purpose. Never use these words: core, mixed, rising, deep cut, mood, picks, favorites, selection, collection, essentials, hits, vibes. Name the most DISTINCTIVE constraint rather than summarising every axis. Good: Crime+Drama, GB, 'based on novel or book' -> British Literary Crime. Sci-Fi+Thriller, 'artificial intelligence' -> Rogue AI Thrillers. Documentary, 'true crime' -> True Crime Investigations. Never write a country CODE (GB, US, KR) in a title -- the code belongs in the country field only. Use the adjective: GB -> British, US -> American, KR -> Korean, JP -> Japanese, FR -> French. Bad: Core Favorites, Mixed Dramas, Rising Mysteries, Deep Cuts, Mood Picks, GB Crime Comedies.\n- Each row: title (2-5 words), genres (list of IDs), keywords (list"
                " of strings), country (string or null).\n- IMPORTANT: Keep combinations simple and achievable."
                " Use max 2 genres and max 1-2 keywords per row. Do NOT combine 3+ niche constraints together"
                " (e.g. avoid Documentary + dark comedy + anthology — too niche).\n- Output a JSON array of 5 objects."
                + avoid_clause
            )

            data = await gemini_service.generate_structured_async(
                prompt=prompt,
                response_schema=list[LLMRowTheme],
                system_instruction=(
                    "You are a creative film curator. Design 5 catalog rows from the user's interest summary."
                    " Row 1 (The Core): strong match. Row 2 (Mixed): blend + variety. Row 3 (Rising Star):"
                    " discovery—suggest new content they would enjoy, not just more of the same. Use genres,"
                    " keywords, and country. Output valid JSON only."
                ),
                api_key=api_key,
            )

            if not data or not isinstance(data, list):
                # Previously returned silently, making an LLM/schema failure
                # indistinguishable from "no key configured" -- both just fell through
                # to tiered sampling with no explanation.
                logger.warning(
                    f"[LLM Rows] unusable response for {content_type}: "
                    f"type={type(data).__name__} value={str(data)[:300]}"
                )
                return None

            final_rows = []
            profile_kw_map = {name.lower(): kid for kid, name in features.keyword_names.items()}

            # Track used anchor genres/keywords across rows to prevent overlapping discover pools
            used_anchor_genres: set[int] = set()
            used_anchor_keywords: set[int] = set()

            for item in data:
                if isinstance(item, dict):
                    title = item.get("title", "Recommended")
                    genre_ids = item.get("genres", [])
                    kw_names = item.get("keywords", [])
                    country = item.get("country")
                else:
                    title = item.title
                    genre_ids = item.genres
                    kw_names = item.keywords
                    country = item.country

                builder = RowBuilder(features)

                # Only add genres not already anchored in previous rows
                row_anchor_genres = []
                for gid in genre_ids:
                    gid = int(gid)
                    if gid in current_genre_map and gid not in used_anchor_genres:
                        builder.add_axis(AXIS_GENRE, gid, AxisRole.ANCHOR)
                        row_anchor_genres.append(gid)
                    elif gid in current_genre_map:
                        # Demote repeated anchor genres to FLAVOR
                        builder.add_axis(AXIS_GENRE, gid, AxisRole.FLAVOR)

                for kw_name in kw_names:
                    kid = await self._resolve_keyword_to_id(kw_name, profile_kw_map)
                    if kid is not None and kid not in GENERIC_KEYWORD_BLACKLIST and kid not in used_anchor_keywords:
                        builder.add_axis(AXIS_KEYWORD, kid, AxisRole.FLAVOR)

                if country:
                    builder.add_axis(AXIS_COUNTRY, country, AxisRole.FLAVOR)

                row_comp = builder.build()
                if row_comp and row_comp.axes:
                    row_id = build_row_id(row_comp.axes)
                    final_rows.append(RowDefinition(title=title, id=row_id, axes=row_comp.axes))
                    used_anchor_genres.update(row_anchor_genres)
                    used_anchor_keywords.update(
                        a.value for a in row_comp.axes if a.name == AXIS_KEYWORD
                    )

            if final_rows:
                final_rows = await self._repair_thin_rows(final_rows, features, content_type)
            return final_rows if final_rows else None

        except Exception as e:
            logger.warning(f"Error in _generate_rows_with_llm: {e}")
            return None
