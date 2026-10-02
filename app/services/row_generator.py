"""
Dynamic Row Generator Service.

Generates 3 personalized catalog rows using a tiered sampling system:
- Row 1 (The Core): User's strongest preferences (Gold tier: Top 1-3)
- Row 2 (The Blend): Mixed preferences with higher complexity (Gold+Silver: Top 1-8)
- Row 3 (The Rising Star): Emerging interests (Silver tier: Rank 4-10)
"""

import asyncio
import hashlib
import json
import time
import random
from enum import Enum
from typing import Any, ClassVar

from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.taste_profile import TasteProfile
from app.services.openrouter import gemini_service
from app.services.tmdb.countries import COUNTRY_ADJECTIVES
from app.services.tmdb.genre import movie_genres, series_genres
from app.services.recommendation.filtering import RecommendationFiltering
from app.services.recommendation.utils import apply_discover_filters
from app.services.tmdb.service import TMDBService, get_tmdb_service

GOLD_TIER_LIMIT = 3  # Top 1-3 items
SILVER_TIER_START = 3  # Rank 4+
SILVER_TIER_END = 10  # Up to Rank 10

# Theme Rotation V2 deliberately keeps a much deeper taste pool available
# for controlled rotation. The legacy algorithm must retain its historical
# 20-candidate fetch -> blacklist -> first-10 behavior when V2 is disabled.
THEME_ROTATION_V2_PROFILE_KEYWORD_LIMIT = 50
THEME_ROTATION_V2_PROMPT_KEYWORD_LIMIT = 12

# V2 deliberately asks for spare themes in the same LLM request. Inventory
# validation can then discard an unusably thin candidate without publishing
# fewer than five rows or spending a second AI request.
THEME_ROTATION_V2_CANDIDATE_COUNT = 12
THEME_ROTATION_V2_PUBLISH_COUNT = 5

# V2 row names should read like customer-facing streaming shelves, not exposed
# metadata filters. The same guidance is used for initial candidate naming and
# repaired-row retitling so a repair cannot regress an expressive title back to
# a mechanical genre/keyword label.
THEME_ROTATION_V2_TITLE_STYLE_GUIDANCE = (
    "Titles are customer-facing streaming shelf names, not database filter "
    "summaries. Keep each title short (2-5 words), natural, memorable, and "
    "specific enough to communicate the row's appeal. Every title must retain "
    "a recognizable hook from the surviving filters so a viewer has some sense "
    "of why these titles belong together. If the same shelf name could plausibly "
    "fit several unrelated themes, it is too vague and must be rewritten. "
    "The finished name must sound like a curated streaming shelf or collection, "
    "not like the title of one individual movie or TV series. Avoid standalone-"
    "title phrasing that could reasonably be mistaken for a single work. For "
    "example, names such as 'Edge of Tomorrow' or 'Cold Case Files' should be "
    "reframed into collection-style wording while preserving the same supported "
    "theme. This rule should improve shelf identity without forcing generic words "
    "such as collection, picks, favorites, or selection into the title. "
    "Do not obscure a defining viewer-facing format just for the sake of being "
    "clever. In particular, when anthology is a surviving keyword, the finished "
    "title MUST contain the word 'Anthology' or 'Anthologies' so viewers can "
    "immediately recognize that the entire shelf consists of anthology content. "
    "Conversely, NEVER use the word 'Anthology' or 'Anthologies' unless anthology "
    "is actually a surviving keyword for the final row. A shelf title must never "
    "imply that every item is anthology content when the filters do not establish "
    "that fact. "
    "But explicit format clarity must not flatten the title into a bare category "
    "label. Keep the identifying format word while giving the shelf personality, "
    "energy, and a memorable hook grounded in the remaining filters. For example, "
    "plain 'Sci-Fi Anthologies' is technically clear but too generic; a title "
    "such as 'Anthologies Beyond Tomorrow' better preserves both clarity and "
    "curated-shelf character when the surviving filters support science fiction. "
    "If repair removes a distinctive keyword that inspired the original title, "
    "never keep that unsupported concept merely because the old wording sounded "
    "better. Instead, preserve the original title's energy using imagery or "
    "phrasing that is genuinely supported by the final surviving filters. "
    "Prefer a concrete image, tension, action, relationship, format, or idea "
    "genuinely implied by the filters. Prefer vivid or clever phrasing when the "
    "filters genuinely support it. Do not mechanically concatenate genre and "
    "keyword terms, stack near-synonyms, or produce clinical labels such as "
    "'Crime Murder Thrillers', 'Atmospheric Sci-Fi', or 'Procedural Crime "
    "Investigations'. Do not hide the actual theme behind empty abstraction such "
    "as 'Unraveling the Unknown' or 'Unfolding Real Stories', and do not default "
    "to an adjective-plus-format construction such as 'Dramatic Anthology Tales'. "
    "Words such as stories, tales, worlds, journeys, unknown, unfolding, or "
    "unraveling are not forbidden, but they must add a specific supported idea "
    "rather than act as generic filler. Avoid generic planning words such as "
    "picks, favorites, selection, collection, essentials, hits, vibes, core, "
    "mixed, rising, deep cut, or mood. A playful, idiomatic, or metaphorical "
    "title is welcome only when every implied concept is grounded in the row's "
    "surviving genre/keyword filters. Use varied imagery, structure, and wording "
    "across neighboring shelves rather than treating any example phrase as a "
    "reusable title template. Never invent "
    "geography, nationality, language, production country, franchise, era, or "
    "subject matter that is not supported by the filters."
)

THEME_ROTATION_V2_RETITLE_SYSTEM_INSTRUCTION = (
    "You are an expert streaming-service catalog editor. Rename the shelf from "
    "its final post-repair filters. Return one title only, with no explanation. "
    + THEME_ROTATION_V2_TITLE_STYLE_GUIDANCE
)

# V2 converts provider watched IDs to one TMDB namespace so inventory probes can
# subtract watched titles cheaply. The normalized result is cached separately
# from legacy watched data and automatically invalidates when the source sets
# change.
THEME_ROTATION_V2_WATCHED_TMDB_TTL_SECONDS = 30 * 24 * 60 * 60
THEME_ROTATION_V2_WATCHED_RESOLVE_CONCURRENCY = 12

# A normal themed shelf shows about 20 titles. V2 aims for substantially more
# unseen inventory so a row remains useful as the user keeps watching.
THEME_ROTATION_V2_UNSEEN_TARGET = 35
THEME_ROTATION_V2_UNSEEN_HARD_FLOOR = 25
THEME_ROTATION_V2_INVENTORY_MAX_PAGES = 10

# ThemeBasedService already relaxes a narrow row's vote-count floor to 10 when
# the normal quality floor leaves fewer than 40 candidates. Mirror that here so
# V2 validates the inventory the production theme service can actually surface.
THEME_ROTATION_V2_NARROW_POOL_THRESHOLD = 40
THEME_ROTATION_V2_RELAXED_VOTE_FLOOR = 10

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


# Minimum raw eligible inventory a themed row must have before it is worth showing.
# The probe applies the user's discover filters and the same AND semantics as the
# real catalog query. Watched-history and final metadata/language filtering happen
# later, so require headroom above the 20-item catalog limit. A target of 25 keeps
# specific themes when they have enough depth while forcing thin 8-15 item rows
# to relax coherently instead of publishing half-full shelves.
MIN_ROW_INVENTORY = 25

# How long a generated row set stays usable before we spend LLM requests on a new
# one. Rows were previously regenerated on every manifest rebuild (every 6h, per
# content type, per profile), which is what exhausts free-tier quotas. A day gives
# visibly fresh themes at 2 requests/day instead of 8+.
ROW_SET_MAX_AGE_SECONDS = 86400

# Keywords too semantically empty to define a row. IDs verified against the TMDB
# keyword endpoint -- four of the original eight comments named the wrong keyword
# ("based on true story" was actually 818 = based on novel or book, "philosophical"
# was 4344 = musical, "suspense" was 663 = fortune teller, "based on novel" was
# 9717 = based on comic), which suppressed several perfectly good themes.
#
# Thin-but-meaningful keywords no longer need to be listed here: MIN_ROW_INVENTORY
# in _repair_thin_rows rejects anything without enough titles behind it.
GENERIC_KEYWORD_BLACKLIST = {
    239797,  # complex              - subjective, describes no subject matter
    197582,  # mysterious           - subjective, describes no subject matter
    2964,    # future               - too broad; overlaps the sci-fi genre
    179430,  # aftercreditsstinger  - technical credits metadata, not a theme
    179431,  # duringcreditsstinger - technical credits metadata, not a theme
    9663,    # sequel              - structural franchise metadata, not a taste subject
    325765,  # amused              - subjective reaction tag, not a useful taste axis
}

# Previously blacklisted by mistake, now allowed:
#   11162  miniseries             - useful format/theme category
#   818    based on novel or book (8257 titles)
#   9717   based on comic        (851 titles)
#   4344   musical               (3844 titles)
#   663    fortune teller        (79 titles - inventory check gates this one)

class RowGeneratorService:
    # Strong references to in-flight background refreshes; asyncio only holds a
    # weak reference to a bare create_task() result.
    _bg_tasks: set = set()

    # Only intervene when a meaningful title word-family has become a real
    # batch-level pattern. Two uses can be natural; three or more across the
    # movie + series shelves is treated as overuse.
    TITLE_WORD_OVERUSE_THRESHOLD = 3

    # These words describe genre, format, or ordinary shelf grammar. Repeating
    # them can be necessary for clarity and must never trigger cosmetic retitles.
    TITLE_WORD_DIVERSITY_PROTECTED = frozenset({
        "movie",
        "movies",
        "series",
        "show",
        "shows",
        "anthology",
        "anthologies",
        "miniseries",
        "action",
        "adventure",
        "animation",
        "animated",
        "comedy",
        "crime",
        "documentary",
        "documentaries",
        "drama",
        "family",
        "fantasy",
        "history",
        "historical",
        "horror",
        "music",
        "musical",
        "mystery",
        "mysteries",
        "reality",
        "romance",
        "science",
        "fiction",
        "scifi",
        "thriller",
        "western",
    })

    """Generates dynamic, personalized row definitions from a User Taste Profile."""

    def __init__(self, tmdb_service: TMDBService | None = None, user_settings=None):
        self.tmdb_service = tmdb_service or get_tmdb_service()
        # Needed so inventory probes count titles the user will ACTUALLY see.
        # Without the year range and vote floor applied, total_results reports a
        # pool several times larger than the one the real query returns, making
        # any MIN_ROW_INVENTORY threshold guesswork.
        self.user_settings = user_settings

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

        # 2. Try LLM generation if either primary BYOK or direct Google BYOK
        # is present. OpenRouter/Groq remains primary; Google is provider fallback.
        google_api_key = (
            getattr(self.user_settings, "gemini_api_key", None)
            if self.user_settings
            else None
        ) or settings.GEMINI_API_KEY

        if api_key or google_api_key:
            # Reuse a recent set rather than spending a request to regenerate one.
            fresh = await self._get_cached_llm_rows(
                token, content_type, max_age=ROW_SET_MAX_AGE_SECONDS
            )
            if fresh:
                logger.info(
                    f"Reusing {len(fresh)} cached LLM rows for {content_type} "
                    "(under 24h old, no LLM request spent)"
                )
                return fresh

            # Stale-while-revalidate: an expired set is still far better than making
            # the caller wait ~40s for the LLM. Serve it now and refresh in the
            # background so the next request gets the new one. Without this a cold
            # rebuild exceeds the client's read timeout and shows nothing at all.
            stale = await self._get_cached_llm_rows(token, content_type, max_age=None)
            if stale:
                logger.info(
                    f"Serving {len(stale)} stale LLM rows for {content_type} and "
                    "regenerating in the background"
                )
                task = asyncio.create_task(
                    self._regenerate_in_background(profile, features, content_type, api_key, token)
                )
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
                return stale

            try:
                llm_rows = await self._generate_rows_with_llm(
                    profile,
                    features,
                    content_type,
                    api_key,
                    avoid_titles,
                    token=token,
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
        # Keep Theme Rotation V2 completely isolated from the legacy row cache.
        #
        # This is essential for the runtime kill switch: disabling V2 must
        # immediately return to the exact legacy cache namespace rather than
        # continuing to serve a previously generated V2 row set.
        namespace = (
            "watchly:llm_rows_v2"
            if settings.THEME_ROTATION_V2_ENABLED
            else "watchly:llm_rows"
        )
        return f"{namespace}:{token}:{content_type}"

    async def _cache_llm_rows(self, token: str | None, content_type: str, rows: list) -> None:
        """Persist the most recent successful LLM row set as a fallback."""
        if not token or not rows:
            return
        try:
            from app.services.redis_service import redis_service
            payload = json.dumps(
                {
                    "generated_at": time.time(),
                    "rows": [r.model_dump(mode="json") for r in rows],
                }
            )
            await redis_service.set(self._llm_rows_key(token, content_type), payload, 604800)
        except Exception as e:
            logger.debug(f"[LLM Rows] failed to cache rows: {e}")

    async def _get_cached_llm_rows(
        self, token: str | None, content_type: str, max_age: float | None = None
    ) -> list | None:
        """Return the last successful LLM row set, if one is stored."""
        if not token:
            return None
        try:
            from app.services.redis_service import redis_service
            raw = await redis_service.get(self._llm_rows_key(token, content_type))
            if not raw:
                return None
            data = json.loads(raw)
            if isinstance(data, list):  # legacy format, no timestamp
                return [RowDefinition(**d) for d in data]
            rows = [RowDefinition(**d) for d in data.get("rows", [])]
            if max_age is not None:
                age = time.time() - float(data.get("generated_at", 0))
                if age > max_age:
                    logger.info(
                        f"[LLM Rows] cached set for {content_type} is "
                        f"{age / 3600:.1f}h old; regenerating"
                    )
                    return None
            return rows
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

        # Remove objectively useless TMDB metadata before it can enter any row
        # generation path.
        #
        # LEGACY:
        #   Preserve the exact historical 20 -> blacklist -> first 10 behavior.
        #
        # V2:
        #   Keep a deeper ranked pool available so later rotation logic can choose
        # fresh-but-still-personalized signals instead of repeatedly feeding the
        # same ten keywords to the LLM. Merely enabling the deeper pool does not
        # itself choose or record rotation history.
        if settings.THEME_ROTATION_V2_ENABLED:
            raw_keywords = profile.get_top_keywords(
                limit=THEME_ROTATION_V2_PROFILE_KEYWORD_LIMIT
            )
            keywords = [
                item
                for item in raw_keywords
                if item[0] not in GENERIC_KEYWORD_BLACKLIST
            ]
        else:
            raw_keywords = profile.get_top_keywords(limit=20)
            keywords = [
                item
                for item in raw_keywords
                if item[0] not in GENERIC_KEYWORD_BLACKLIST
            ][:10]

        countries = []
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

        # 2. Flavor: Secondary Genre
        # Production country is not a taste signal.
        other_genres = [g for g in features.genres if g[0] != genres[0][0]]
        if other_genres:
            sec_genre = sample_from_gold_silver(other_genres, 1)
            builder.add_axis(AXIS_GENRE, sec_genre[0][0], AxisRole.FLAVOR, 0.7)

        row = builder.build()
        if row:
            row.explanation = "The Blend: Mixing your top genres with complementary secondary interests."
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

    @staticmethod
    def _title_mentions_anthology(
        title: str,
    ) -> bool:
        """Return whether visible wording claims anthology content."""
        normalized = "".join(
            char.casefold()
            if char.isalnum()
            else " "
            for char in str(title or "")
        )

        return any(
            token.startswith("antholog")
            for token in normalized.split()
        )

    @classmethod
    def _row_has_anthology_keyword(
        cls,
        row: RowDefinition,
        features: ExtractedFeatures,
    ) -> bool:
        """Return whether anthology survives in the final keyword axes."""
        for axis in row.axes:
            if axis.name != AXIS_KEYWORD:
                continue

            try:
                keyword_id = int(axis.value)
            except (TypeError, ValueError):
                continue

            keyword_name = (
                features.get_keyword_name(keyword_id)
                or ""
            )

            if cls._title_mentions_anthology(
                keyword_name
            ):
                return True

        return False

    async def _generate_titles(self, rows_data: list[RowComponents]) -> list[RowDefinition]:
        """Generate titles using the user's BYOK key when configured."""
        if not rows_data:
            return []

        api_key = (
            getattr(self.user_settings, "openrouter_api_key", None)
            if self.user_settings
            else None
        )

        prompts = [row.build_prompt() for row in rows_data]
        gemini_tasks = [
            gemini_service.generate_content_async(
                prompt,
                api_key=api_key,
                google_api_key=(
                    (
                        getattr(
                            self.user_settings,
                            "gemini_api_key",
                            None,
                        )
                        if self.user_settings
                        else None
                    )
                    or settings.GEMINI_API_KEY
                ),
            )
            for prompt in prompts
        ]
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

    async def _retitle_repaired_rows(
        self,
        rows: list[RowDefinition],
        features: ExtractedFeatures,
    ) -> None:
        """Rename repaired rows from their final axes using the normal title AI.

        Repair can add, remove, or replace keyword axes after the original title
        was generated. Rebuild the naming prompt from the final axes so stale or
        mechanically concatenated titles are never published. If title generation
        fails, keep the previous title rather than degrading to a raw fallback.
        """
        if not rows:
            return

        api_key = (
            getattr(self.user_settings, "openrouter_api_key", None)
            if self.user_settings
            else None
        )

        prompts = []
        for row in rows:
            builder = RowBuilder(features)
            for axis in row.axes:
                builder.add_axis(axis.name, axis.value, axis.role, axis.weight)
            final_filters = builder.components.build_prompt()
            prompts.append(
                f"Existing title: {row.title}\n"
                f"Final row filters after repair:\n{final_filters}\n\n"
                "Rename this customer-facing shelf from the FINAL filters. Preserve "
                "the existing title only if it is both accurate AND already polished, "
                "natural, and memorable. If it is literal, generic, repetitive, or "
                "reads like exposed metadata, rewrite it even when technically accurate. "
                "Every concrete subject or theme in the title must be grounded in a "
                "surviving genre or keyword. If a removed keyword was the only support "
                "for a concept such as psychological, noir, consultant, heist, hitman, "
                "superhero, or period, remove or replace that concept. Tone-only "
                "modifiers may remain only when they reasonably describe the surviving "
                "filters. "
                + THEME_ROTATION_V2_TITLE_STYLE_GUIDANCE
            )

        tasks = [
            gemini_service.generate_content_async(
                prompt,
                api_key=api_key,
                google_api_key=(
                    (
                        getattr(
                            self.user_settings,
                            "gemini_api_key",
                            None,
                        )
                        if self.user_settings
                        else None
                    )
                    or settings.GEMINI_API_KEY
                ),
                system_instruction=(
                    THEME_ROTATION_V2_RETITLE_SYSTEM_INSTRUCTION
                    if settings.THEME_ROTATION_V2_ENABLED
                    else None
                ),
            )
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for row, result in zip(rows, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[RowRepair] AI retitle failed for '{row.title}': {result}"
                )
                continue
            if not result:
                logger.warning(
                    f"[RowRepair] AI retitle returned empty for '{row.title}'; "
                    "keeping previous title"
                )
                continue

            old_title = row.title
            row.title = self._clean_title(result)
            logger.info(
                f"[RowRepair] AI retitle '{old_title}' -> '{row.title}'"
            )

    @classmethod
    def _title_content_word_families(
        cls,
        title: str,
    ) -> set[str]:
        """Return meaningful lexical families used by a visible shelf title.

        This is intentionally conservative. Short grammar words and explicit
        genre/format terms are ignored. Simple plural variants collapse into the
        same family so Future/Futures, World/Worlds, Story/Stories, etc. can be
        recognized as the same repeated wording.
        """
        normalized = "".join(
            char.casefold()
            if char.isalnum()
            else " "
            for char in str(title or "")
        )

        families: set[str] = set()

        for token in normalized.split():
            if len(token) < 5:
                continue

            if token in cls.TITLE_WORD_DIVERSITY_PROTECTED:
                continue

            family = token

            if (
                token.endswith("ies")
                and len(token) > 5
                and token != "series"
            ):
                family = token[:-3] + "y"
            elif (
                token.endswith("s")
                and len(token) > 5
                and not token.endswith(("ss", "us", "is"))
            ):
                family = token[:-1]

            if (
                len(family) < 5
                or family in cls.TITLE_WORD_DIVERSITY_PROTECTED
            ):
                continue

            families.add(family)

        return families

    async def _retitle_for_word_diversity(
        self,
        row: RowDefinition,
        features: ExtractedFeatures,
        avoid_families: set[str],
    ) -> str | None:
        """Cosmetically rename a valid row without changing its recipe.

        Failure is deliberately non-destructive: the caller keeps the existing
        title and the row itself is never rejected because of lexical repetition.
        """
        if not avoid_families:
            return None

        builder = RowBuilder(features)

        for axis in row.axes:
            builder.add_axis(
                axis.name,
                axis.value,
                axis.role,
                axis.weight,
            )

        final_filters = builder.components.build_prompt()
        avoid_text = ", ".join(sorted(avoid_families))

        prompt = (
            f"Existing title: {row.title}\n"
            f"Final row filters:\n{final_filters}\n\n"
            "This shelf is already valid. Rename ONLY to remove repeated wording "
            "that is overused across neighboring shelves. Do not change what the "
            "shelf means and do not change or imply any filters. "
            f"Avoid these overused word families: {avoid_text}. "
            "Do not use those words or simple singular/plural variants of them. "
            "Find fresh wording genuinely supported by the FINAL filters. "
            "Every concrete idea in the replacement must be grounded in those "
            "filters. Return one title only, with no explanation. "
            + THEME_ROTATION_V2_TITLE_STYLE_GUIDANCE
        )

        api_key = (
            getattr(
                self.user_settings,
                "openrouter_api_key",
                None,
            )
            if self.user_settings
            else None
        )

        try:
            result = await gemini_service.generate_content_async(
                prompt,
                api_key=api_key,
                google_api_key=(
                    (
                        getattr(
                            self.user_settings,
                            "gemini_api_key",
                            None,
                        )
                        if self.user_settings
                        else None
                    )
                    or settings.GEMINI_API_KEY
                ),
                system_instruction=(
                    THEME_ROTATION_V2_RETITLE_SYSTEM_INSTRUCTION
                ),
            )
        except Exception as exc:
            logger.warning(
                f"[TitleDiversity] retitle failed for "
                f"'{row.title}': {exc}"
            )
            return None

        if not result:
            logger.warning(
                f"[TitleDiversity] empty retitle for "
                f"'{row.title}'; keeping original"
            )
            return None

        replacement = self._clean_title(result)

        if (
            not replacement
            or replacement.strip().casefold()
            == row.title.strip().casefold()
        ):
            return None

        replacement_families = (
            self._title_content_word_families(
                replacement
            )
        )

        if replacement_families & avoid_families:
            logger.warning(
                f"[TitleDiversity] replacement still used "
                f"overused wording for '{row.title}': "
                f"'{replacement}'"
            )
            return None

        return replacement

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

    @staticmethod
    def _v2_normalized_watched_key(
        token: str,
        content_type: str,
    ) -> str:
        return (
            f"watchly:theme_rotation_v2_watched_tmdb:"
            f"{token}:{content_type}"
        )

    @staticmethod
    def _v2_watched_signature(
        watched_tmdb: set[int],
        watched_imdb: set[str],
    ) -> str:
        """Build a stable signature of the provider watched sets."""
        tmdb_part = ",".join(
            str(value)
            for value in sorted(
                int(value)
                for value in watched_tmdb
            )
        )

        imdb_part = ",".join(
            sorted(
                str(value).strip().lower()
                for value in watched_imdb
                if str(value).strip()
            )
        )

        payload = (
            f"tmdb={tmdb_part}\n"
            f"imdb={imdb_part}"
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()

    async def _resolve_v2_watched_tmdb_ids(
        self,
        watched_tmdb: set[int],
        watched_imdb: set[str],
        content_type: str,
    ) -> set[int]:
        """Normalize provider watched IDs into TMDB IDs.

        Trakt/Stremio may already supply TMDB IDs. Simkl commonly supplies IMDb
        IDs only. Resolving the watched side once is much cheaper than enriching
        every candidate row title just to discover its IMDb ID.
        """
        normalized: set[int] = set()

        for value in watched_tmdb or set():
            try:
                normalized.add(int(value))
            except (TypeError, ValueError):
                continue

        imdb_ids = sorted({
            str(value).strip().lower()
            for value in (watched_imdb or set())
            if str(value).strip()
        })

        if not imdb_ids:
            return normalized

        expected_tmdb_type = (
            "movie"
            if content_type == "movie"
            else "tv"
        )

        semaphore = asyncio.Semaphore(
            THEME_ROTATION_V2_WATCHED_RESOLVE_CONCURRENCY
        )

        async def resolve_one(
            imdb_id: str,
        ) -> int | None:
            async with semaphore:
                try:
                    tmdb_id, resolved_type = (
                        await self.tmdb_service.find_by_imdb_id(
                            imdb_id
                        )
                    )
                except Exception as exc:
                    logger.debug(
                        f"[ThemeRotationV2] IMDb->TMDB lookup failed "
                        f"for {imdb_id}: {exc}"
                    )
                    return None

            if not tmdb_id:
                return None

            if (
                resolved_type
                and resolved_type != expected_tmdb_type
            ):
                return None

            try:
                return int(tmdb_id)
            except (TypeError, ValueError):
                return None

        resolved = await asyncio.gather(
            *[
                resolve_one(imdb_id)
                for imdb_id in imdb_ids
            ]
        )

        normalized.update(
            tmdb_id
            for tmdb_id in resolved
            if tmdb_id is not None
        )

        return normalized

    async def _get_v2_normalized_watched_tmdb(
        self,
        token: str | None,
        content_type: str,
    ) -> set[int] | None:
        """Return the cached normalized V2 watched-TMDB set.

        None means the source watched sets were unavailable, so callers can
        distinguish "no watched titles" from "could not validate watched state."
        """
        if not token:
            return None

        try:
            from app.services.user_cache import user_cache

            watched_sets = await user_cache.get_watched_sets(
                token,
                content_type,
            )
        except Exception as exc:
            logger.warning(
                f"[ThemeRotationV2] failed to read watched sets for "
                f"{content_type}: {exc}"
            )
            return None

        if watched_sets is None:
            return None

        watched_tmdb_raw, watched_imdb_raw = watched_sets

        watched_tmdb = set()

        for value in watched_tmdb_raw or set():
            try:
                watched_tmdb.add(int(value))
            except (TypeError, ValueError):
                continue

        watched_imdb = {
            str(value).strip().lower()
            for value in (watched_imdb_raw or set())
            if str(value).strip()
        }

        signature = self._v2_watched_signature(
            watched_tmdb,
            watched_imdb,
        )

        cache_key = self._v2_normalized_watched_key(
            token,
            content_type,
        )

        try:
            from app.services.redis_service import redis_service

            raw = await redis_service.get(
                cache_key
            )

            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")

                data = json.loads(raw)

                if (
                    isinstance(data, dict)
                    and data.get("signature") == signature
                ):
                    cached_ids = {
                        int(value)
                        for value in (
                            data.get("tmdb_ids")
                            or []
                        )
                    }

                    logger.debug(
                        f"[ThemeRotationV2] using cached normalized "
                        f"watched set for {content_type}: "
                        f"{len(cached_ids)} TMDB IDs"
                    )

                    return cached_ids

        except Exception as exc:
            logger.debug(
                f"[ThemeRotationV2] normalized watched cache read "
                f"failed for {content_type}: {exc}"
            )

        normalized = (
            await self._resolve_v2_watched_tmdb_ids(
                watched_tmdb,
                watched_imdb,
                content_type,
            )
        )

        try:
            payload = json.dumps({
                "signature": signature,
                "tmdb_ids": sorted(normalized),
                "source_tmdb_count": len(
                    watched_tmdb
                ),
                "source_imdb_count": len(
                    watched_imdb
                ),
            })

            await redis_service.set(
                cache_key,
                payload,
                THEME_ROTATION_V2_WATCHED_TMDB_TTL_SECONDS,
            )

        except Exception as exc:
            # Cache failure must not discard the valid in-memory normalized set.
            logger.debug(
                f"[ThemeRotationV2] normalized watched cache write "
                f"failed for {content_type}: {exc}"
            )

        logger.info(
            f"[ThemeRotationV2] normalized watched history for "
            f"{content_type}: "
            f"tmdb_source={len(watched_tmdb)} "
            f"imdb_source={len(watched_imdb)} "
            f"normalized_tmdb={len(normalized)}"
        )

        return normalized

    def _v2_inventory_params(
        self,
        axes: list,
        content_type: str,
    ) -> dict[str, Any]:
        """Build the effective discover query for a V2 inventory check."""
        genres = [
            str(axis.value)
            for axis in axes
            if axis.name == AXIS_GENRE
        ]

        keywords = [
            str(axis.value)
            for axis in axes
            if axis.name == AXIS_KEYWORD
        ]

        countries = [
            str(axis.value)
            for axis in axes
            if axis.name == AXIS_COUNTRY
        ]

        params: dict[str, Any] = {}

        if genres:
            params["with_genres"] = ",".join(
                genres
            )

        if keywords:
            params["with_keywords"] = ",".join(
                keywords
            )

        if countries:
            params["with_origin_country"] = (
                countries[0]
            )

        if not params:
            return {}

        if self.user_settings is not None:
            params = apply_discover_filters(
                params,
                self.user_settings,
            )

            excluded_ids = (
                RecommendationFiltering
                .get_excluded_genre_ids(
                    self.user_settings,
                    content_type,
                )
            )

            if excluded_ids:
                included_genres = set()

                for value in (
                    params.get(
                        "with_genres",
                        "",
                    )
                    or ""
                ).replace("|", ",").split(","):
                    if not value:
                        continue

                    try:
                        included_genres.add(
                            int(value)
                        )
                    except ValueError:
                        continue

                without = [
                    genre_id
                    for genre_id
                    in excluded_ids
                    if genre_id
                    not in included_genres
                ]

                if without:
                    params["without_genres"] = (
                        "|".join(
                            str(genre_id)
                            for genre_id
                            in without
                        )
                    )

        return params

    async def _v2_row_unseen_inventory(
        self,
        axes: list,
        content_type: str,
        watched_tmdb: set[int],
        target: int = THEME_ROTATION_V2_UNSEEN_TARGET,
        hard_floor: int = THEME_ROTATION_V2_UNSEEN_HARD_FLOOR,
    ) -> dict[str, Any]:
        """Measure genuinely unseen inventory for one proposed V2 row.

        The probe uses the actual discover constraints and user filters, removes
        watched TMDB IDs, and stops early once the healthy target is proven.

        A transient TMDB failure returns validated=False rather than pretending
        the row has zero inventory. Later acceptance logic can then fail open to
        the existing legacy/raw behavior instead of destroying a good row during
        an upstream outage.
        """
        params = self._v2_inventory_params(
            axes,
            content_type,
        )

        empty_result = {
            "validated": True,
            "raw_total": 0,
            "unseen_count": 0,
            "watched_removed": 0,
            "candidates_scanned": 0,
            "target_met": False,
            "hard_floor_met": False,
            "complete": True,
            "scan_capped": False,
            "vote_floor_relaxed": False,
        }

        if not params:
            return empty_result

        watched = set()

        for value in watched_tmdb or set():
            try:
                watched.add(int(value))
            except (TypeError, ValueError):
                continue

        try:
            first_page = (
                await self.tmdb_service.get_discover(
                    content_type,
                    page=1,
                    **params,
                )
            )

            raw_total = int(
                first_page.get(
                    "total_results",
                    0,
                )
                or 0
            )

            vote_floor_relaxed = False

            floor = params.get(
                "vote_count.gte"
            )

            try:
                floor_int = int(floor)
            except (
                TypeError,
                ValueError,
            ):
                floor_int = 0

            if (
                raw_total
                < THEME_ROTATION_V2_NARROW_POOL_THRESHOLD
                and floor_int
                > THEME_ROTATION_V2_RELAXED_VOTE_FLOOR
            ):
                relaxed_params = dict(
                    params
                )

                relaxed_params[
                    "vote_count.gte"
                ] = (
                    THEME_ROTATION_V2_RELAXED_VOTE_FLOOR
                )

                wider_first = (
                    await self.tmdb_service.get_discover(
                        content_type,
                        page=1,
                        **relaxed_params,
                    )
                )

                wider_total = int(
                    wider_first.get(
                        "total_results",
                        0,
                    )
                    or 0
                )

                if wider_total > raw_total:
                    params = relaxed_params
                    first_page = wider_first
                    raw_total = wider_total
                    vote_floor_relaxed = True

            total_pages = int(
                first_page.get(
                    "total_pages",
                    0,
                )
                or 0
            )

            seen_ids: set[int] = set()
            unseen_ids: set[int] = set()

            watched_removed = 0
            scanned = 0

            async def consume(
                data: dict[str, Any],
            ) -> None:
                nonlocal watched_removed
                nonlocal scanned

                for item in (
                    data.get("results")
                    or []
                ):
                    try:
                        tmdb_id = int(
                            item.get("id")
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    if tmdb_id in seen_ids:
                        continue

                    seen_ids.add(tmdb_id)
                    scanned += 1

                    if tmdb_id in watched:
                        watched_removed += 1
                        continue

                    unseen_ids.add(
                        tmdb_id
                    )

                    if len(unseen_ids) >= target:
                        return

            await consume(
                first_page
            )

            last_page_scanned = 1

            page_limit = min(
                max(
                    total_pages,
                    1,
                ),
                THEME_ROTATION_V2_INVENTORY_MAX_PAGES,
            )

            for page in range(
                2,
                page_limit + 1,
            ):
                if len(unseen_ids) >= target:
                    break

                data = (
                    await self.tmdb_service.get_discover(
                        content_type,
                        page=page,
                        **params,
                    )
                )

                last_page_scanned = page

                results = (
                    data.get("results")
                    or []
                )

                if not results:
                    break

                await consume(data)

            target_met = (
                len(unseen_ids) >= target
            )

            complete = (
                not target_met
                and total_pages <= last_page_scanned
            )

            scan_capped = (
                not target_met
                and total_pages > last_page_scanned
                and last_page_scanned
                >= THEME_ROTATION_V2_INVENTORY_MAX_PAGES
            )

            return {
                "validated": True,
                "raw_total": raw_total,
                "unseen_count": len(
                    unseen_ids
                ),
                "watched_removed":
                    watched_removed,
                "candidates_scanned":
                    scanned,
                "target_met":
                    target_met,
                "hard_floor_met":
                    (
                        len(unseen_ids)
                        >= hard_floor
                    ),
                "complete":
                    complete,
                "scan_capped":
                    scan_capped,
                "vote_floor_relaxed":
                    vote_floor_relaxed,
            }

        except Exception as exc:
            logger.warning(
                f"[ThemeRotationV2] unseen inventory probe failed "
                f"for {content_type}: {exc}"
            )

            return {
                "validated": False,
                "raw_total": None,
                "unseen_count": None,
                "watched_removed": None,
                "candidates_scanned": None,
                "target_met": None,
                "hard_floor_met": None,
                "complete": False,
                "scan_capped": False,
                "vote_floor_relaxed": False,
            }

    @staticmethod
    def _v2_axes_signature(
        axes: list[RowAxis],
    ) -> tuple[tuple[str, str], ...]:
        """Inventory-equivalent signature for deduplicating repair attempts."""
        return tuple(
            sorted({
                (
                    str(axis.name),
                    str(axis.value),
                )
                for axis in axes
            })
        )

    async def _v2_repair_row_for_unseen(
        self,
        row: RowDefinition,
        content_type: str,
        watched_tmdb: set[int],
    ) -> dict[str, Any]:
        """Try to make one V2 theme healthy without replacing its concept.

        Policy:
        - >=35 unseen: accept unchanged.
        - 25-34 unseen: try to broaden to >=35; if no better repair exists,
          keep the original because it remains above the hard floor.
        - <25 unseen: repair is mandatory. If no grounded repair reaches 25,
          reject the theme.
        - Probe/network uncertainty fails open; never destroy a row because
          TMDB was temporarily unavailable.

        Repairs only remove constraints already present in the row. They never
        inject an unrelated profile keyword.
        """
        working = row.model_copy(
            deep=True
        )

        working.axes = (
            self._normalize_row_axes(
                list(working.axes)
            )
        )

        working.id = build_row_id(
            working.axes
        )

        original_probe = (
            await self._v2_row_unseen_inventory(
                working.axes,
                content_type,
                watched_tmdb,
            )
        )

        base = {
            "row": working,
            "probe": original_probe,
            "changed": False,
            "original_probe":
                original_probe,
            "attempts": [],
        }

        if not original_probe.get(
            "validated"
        ):
            return {
                **base,
                "status": "unvalidated-keep",
            }

        if original_probe.get(
            "target_met"
        ):
            return {
                **base,
                "status": "healthy",
            }

        # If the probe hit our page safety cap before proving even the hard
        # floor, the result is uncertain rather than a valid rejection.
        if (
            original_probe.get(
                "scan_capped"
            )
            and not original_probe.get(
                "hard_floor_met"
            )
        ):
            return {
                **base,
                "status":
                    "scan-capped-keep",
            }

        original_floor_met = bool(
            original_probe.get(
                "hard_floor_met"
            )
        )

        original_axes = list(
            working.axes
        )

        keyword_axes = [
            axis
            for axis in original_axes
            if axis.name == AXIS_KEYWORD
        ]

        genre_axes = [
            axis
            for axis in original_axes
            if axis.name == AXIS_GENRE
        ]

        candidates: list[
            tuple[
                str,
                list[RowAxis],
            ]
        ] = []

        #
        # 1. Relax one genre while preserving all of the theme's keywords.
        #
        if (
            keyword_axes
            and len(genre_axes) > 1
        ):
            removable = sorted(
                genre_axes,
                key=lambda axis:
                    0
                    if axis.role
                    == AxisRole.FLAVOR
                    else 1,
            )

            for genre_axis in removable:
                candidate = []
                removed = False

                for axis in original_axes:
                    if (
                        not removed
                        and axis is genre_axis
                    ):
                        removed = True
                        continue

                    candidate.append(
                        axis.model_copy(
                            deep=True
                        )
                    )

                if candidate:
                    candidates.append((
                        "relax-one-genre",
                        candidate,
                    ))

        #
        # 2. If multiple keywords were combined, try each original keyword
        #    individually while retaining the non-keyword constraints.
        #
        if len(keyword_axes) > 1:
            non_keyword = [
                axis
                for axis in original_axes
                if axis.name
                != AXIS_KEYWORD
            ]

            for keyword_axis in keyword_axes:
                candidates.append((
                    "single-original-keyword",
                    [
                        *[
                            axis.model_copy(
                                deep=True
                            )
                            for axis
                            in non_keyword
                        ],
                        keyword_axis.model_copy(
                            deep=True
                        ),
                    ],
                ))

        #
        # 3. Preserve one original keyword with one original genre.
        #
        if keyword_axes and genre_axes:
            non_genre_non_keyword = [
                axis
                for axis in original_axes
                if axis.name
                not in (
                    AXIS_GENRE,
                    AXIS_KEYWORD,
                )
            ]

            for keyword_axis in keyword_axes:
                for genre_axis in genre_axes:
                    candidates.append((
                        "keyword-plus-one-genre",
                        [
                            *[
                                axis.model_copy(
                                    deep=True
                                )
                                for axis
                                in non_genre_non_keyword
                            ],
                            genre_axis.model_copy(
                                deep=True
                            ),
                            keyword_axis.model_copy(
                                deep=True
                            ),
                        ],
                    ))

        #
        # 4. Last grounded broadening: keep the original keyword itself and
        #    discard genre narrowing. This preserves the distinctive theme
        #    instead of falling back to a generic broad genre shelf.
        #
        for keyword_axis in keyword_axes:
            other_constraints = [
                axis
                for axis in original_axes
                if axis.name
                not in (
                    AXIS_GENRE,
                    AXIS_KEYWORD,
                )
            ]

            candidates.append((
                "keyword-only",
                [
                    *[
                        axis.model_copy(
                            deep=True
                        )
                        for axis
                        in other_constraints
                    ],
                    keyword_axis.model_copy(
                        deep=True
                    ),
                ],
            ))

        seen_signatures = {
            self._v2_axes_signature(
                original_axes
            )
        }

        best_floor_row = None
        best_floor_probe = None
        best_floor_reason = None

        attempts = []

        for reason, candidate_axes in candidates:
            candidate_axes = (
                self._normalize_row_axes(
                    candidate_axes
                )
            )

            signature = (
                self._v2_axes_signature(
                    candidate_axes
                )
            )

            if (
                not candidate_axes
                or signature
                in seen_signatures
            ):
                continue

            seen_signatures.add(
                signature
            )

            probe = (
                await self._v2_row_unseen_inventory(
                    candidate_axes,
                    content_type,
                    watched_tmdb,
                )
            )

            attempts.append({
                "reason": reason,
                "axes": [
                    {
                        "name": axis.name,
                        "value": axis.value,
                        "role": axis.role.value,
                    }
                    for axis in candidate_axes
                ],
                "probe": probe,
            })

            if not probe.get(
                "validated"
            ):
                continue

            candidate = (
                working.model_copy(
                    deep=True
                )
            )

            candidate.axes = (
                candidate_axes
            )

            candidate.id = build_row_id(
                candidate_axes
            )

            if probe.get(
                "target_met"
            ):
                return {
                    "status":
                        "repaired-target",
                    "row": candidate,
                    "probe": probe,
                    "changed": True,
                    "repair_reason":
                        reason,
                    "original_probe":
                        original_probe,
                    "attempts":
                        attempts,
                }

            if probe.get(
                "hard_floor_met"
            ):
                if (
                    best_floor_probe is None
                    or int(
                        probe.get(
                            "unseen_count"
                        )
                        or 0
                    )
                    > int(
                        best_floor_probe.get(
                            "unseen_count"
                        )
                        or 0
                    )
                ):
                    best_floor_row = (
                        candidate
                    )
                    best_floor_probe = (
                        probe
                    )
                    best_floor_reason = (
                        reason
                    )

        #
        # A 25-34 original row remains publishable. We only change it when the
        # repair actually gets us to the healthy target.
        #
        if original_floor_met:
            return {
                **base,
                "status":
                    "thin-keep",
                "attempts":
                    attempts,
            }

        #
        # Original was below 25. A grounded repair reaching at least 25 is
        # acceptable even if it cannot reach 35.
        #
        if best_floor_row is not None:
            return {
                "status":
                    "repaired-floor",
                "row":
                    best_floor_row,
                "probe":
                    best_floor_probe,
                "changed": True,
                "repair_reason":
                    best_floor_reason,
                "original_probe":
                    original_probe,
                "attempts":
                    attempts,
            }

        #
        # No grounded version of this theme has enough confirmed unseen titles.
        #
        return {
            "status": "reject",
            "row": None,
            "probe":
                original_probe,
            "changed": False,
            "original_probe":
                original_probe,
            "attempts":
                attempts,
        }

    async def _row_inventory(self, axes: list, content_type: str) -> int:
        """Return TMDB's total_results for the discover query a row will run."""
        genres = [str(a.value) for a in axes if a.name == AXIS_GENRE]
        keywords = [str(a.value) for a in axes if a.name == AXIS_KEYWORD]
        countries = [str(a.value) for a in axes if a.name == AXIS_COUNTRY]

        params = {}
        # Must match the semantics of the real catalog query. theme_based.py joins
        # these with "," (AND) -- probing with "|" (OR) counted a far larger pool
        # than the row would actually return, so rows passed the inventory check
        # and then rendered empty.
        if genres:
            params["with_genres"] = ",".join(genres)
        if keywords:
            params["with_keywords"] = ",".join(keywords)
        if countries:
            params["with_origin_country"] = countries[0]
        if not params:
            return 0

        try:
            if self.user_settings is not None:
                params = apply_discover_filters(params, self.user_settings)
            res = await self.tmdb_service.get_discover(content_type, page=1, **params)
            return int(res.get("total_results", 0) or 0)
        except Exception as e:
            logger.debug(f"[RowRepair] inventory probe failed: {e}")
            # Fail open: never discard a row because of a transient TMDB error.
            return MIN_ROW_INVENTORY

    @staticmethod
    def _normalize_row_axes(axes: list[RowAxis]) -> list[RowAxis]:
        # Deduplicate axes and guarantee that a surviving row has an anchor.
        normalized: list[RowAxis] = []
        seen: dict[tuple[str, str], RowAxis] = {}
        role_priority = {
            AxisRole.FALLBACK: 0,
            AxisRole.FLAVOR: 1,
            AxisRole.ANCHOR: 2,
        }

        for axis in axes:
            key = (axis.name, str(axis.value))
            existing = seen.get(key)
            if existing is None:
                normalized.append(axis)
                seen[key] = axis
                continue

            # Keep one copy of an exact axis and preserve its strongest role.
            if role_priority.get(axis.role, 0) > role_priority.get(existing.role, 0):
                existing.role = axis.role
            existing.weight = max(existing.weight, axis.weight)
            if not existing.display and axis.display:
                existing.display = axis.display

        if normalized and not any(a.role == AxisRole.ANCHOR for a in normalized):
            # Repairs may remove the original anchor. Prefer a surviving genre as
            # the replacement anchor because it remains the broadest stable theme.
            promote = next((a for a in normalized if a.name == AXIS_GENRE), None)
            if promote is None:
                promote = next((a for a in normalized if a.name == AXIS_KEYWORD), None)
            if promote is None:
                promote = normalized[0]
            promote.role = AxisRole.ANCHOR

        return normalized

    async def _repair_thin_rows(
        self,
        rows: list,
        features: "ExtractedFeatures",
        content_type: str,
    ) -> list:
        """Ensure every row has enough eligible inventory without changing its theme.

        Repair first relaxes secondary constraints while preserving the row's own
        keyword concepts. It never substitutes an unrelated profile keyword merely
        to hit the inventory floor. If the original keyword cannot support a healthy
        row, the keyword is removed and the title is revised from the surviving axes.
        """
        retitle_rows: list[RowDefinition] = []

        for row in rows:
            # Normalize before probing so legacy/re-anchored duplicate axes do not
            # artificially narrow the TMDB query.
            row.axes = self._normalize_row_axes(list(row.axes))
            row.id = build_row_id(row.axes)
            kw_axes = [a for a in row.axes if a.name == AXIS_KEYWORD]

            # A bare single-axis row is under-specified, but inventing a keyword
            # from elsewhere in the profile changes the theme rather than repairing
            # it. Keep the grounded axis and force a title review instead.
            distinct_axes = {(a.name, str(a.value)) for a in row.axes}
            if not kw_axes and len(distinct_axes) < 2:
                retitle_rows.append(row)
                logger.info(
                    f"[RowRepair] under-specified row -> '{row.title}' "
                    "(kept grounded axes; no unrelated keyword injected)"
                )
                continue

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

            # Step 1: preserve the row's defining keyword theme by relaxing one
            # secondary genre first. Prefer FLAVOR genres; if all genres are
            # anchors, remove only an extra anchor and always leave one genre.
            genre_axes = [a for a in row.axes if a.name == AXIS_GENRE]
            if len(genre_axes) > 1:
                removable_genres = sorted(
                    genre_axes,
                    key=lambda a: 0 if a.role == AxisRole.FLAVOR else 1,
                )
                for genre_axis in removable_genres:
                    candidate = []
                    removed = False
                    for axis in row.axes:
                        if not removed and axis is genre_axis:
                            removed = True
                            continue
                        candidate.append(axis)

                    if not any(a.name == AXIS_GENRE for a in candidate):
                        continue

                    if await self._row_inventory(candidate, content_type) >= MIN_ROW_INVENTORY:
                        row.axes = candidate
                        row.id = build_row_id(candidate)
                        retitle_rows.append(row)
                        repaired = True
                        logger.info(
                            f"[RowRepair] relaxed genre {genre_axis.value} while preserving "
                            f"keyword theme -> '{row.title}'"
                        )
                        break

            if repaired:
                continue

            # Step 2: if several keywords made the row too narrow, keep one of
            # the row's OWN keywords with all surviving non-keyword axes.
            for kw_axis in kw_axes:
                candidate = list(non_kw) + [kw_axis]
                if await self._row_inventory(candidate, content_type) >= MIN_ROW_INVENTORY:
                    row.axes = candidate
                    row.id = build_row_id(candidate)
                    retitle_rows.append(row)
                    repaired = True
                    logger.info(
                        f"[RowRepair] simplified to existing keyword {kw_axis.value} "
                        f"-> '{row.title}'"
                    )
                    break

            if repaired:
                continue

            # Step 3: for multi-keyword rows, try one original keyword with one
            # surviving genre. This broadens the query while preserving the actual
            # subject/theme the LLM chose instead of swapping in an unrelated taste.
            if len(kw_axes) > 1:
                other_non_kw = [a for a in non_kw if a.name != AXIS_GENRE]
                for kw_axis in kw_axes:
                    for genre_axis in genre_axes:
                        candidate = other_non_kw + [genre_axis, kw_axis]
                        if await self._row_inventory(candidate, content_type) >= MIN_ROW_INVENTORY:
                            row.axes = candidate
                            row.id = build_row_id(candidate)
                            retitle_rows.append(row)
                            repaired = True
                            logger.info(
                                f"[RowRepair] broadened to original keyword {kw_axis.value} "
                                f"+ genre {genre_axis.value} -> '{row.title}'"
                            )
                            break
                    if repaired:
                        break

            if repaired:
                continue

            # Step 4: last resort, remove the unsupported keyword constraint and
            # keep the broader grounded axes. Never inject an unrelated keyword.
            if non_kw:
                row.axes = non_kw
                row.id = build_row_id(non_kw)
                retitle_rows.append(row)
                logger.info(f"[RowRepair] dropped unsupported keyword axis -> '{row.title}'")

        # A repair can remove the only anchor (for example, relaxing an anchor
        # genre while leaving a flavor genre + keyword). Normalize once more after
        # all repair decisions so every published row has a valid, deduplicated ID.
        for row in rows:
            row.axes = self._normalize_row_axes(list(row.axes))
            row.id = build_row_id(row.axes)

        await self._retitle_repaired_rows(retitle_rows, features)
        return rows

    async def _select_v2_publishable_rows(
        self,
        rows: list[RowDefinition],
        features: "ExtractedFeatures",
        content_type: str,
        token: str | None,
        history: list[dict[str, Any]] | None = None,
    ) -> list[RowDefinition] | None:
        """Validate V2 candidates and return exactly five publishable rows.

        Candidates are considered in model order. Each one is checked against the
        user's normalized watched history and repaired only by relaxing constraints
        already present in that theme.

        A generation is accepted only when five rows survive. We never cache a
        partial V2 generation merely because some candidates were rejected.

        Cooldown is enforced against the FINAL resolved TMDB keyword axes, not
        merely the prompt hint. This prevents the model from independently
        reintroducing a recently used concept that was intentionally omitted from
        the rotating prompt window.

        Candidate 1 gets one narrow exemption: the user's strongest usable profile
        keyword may repeat as the stable core signal. Every other recent keyword is
        blocked, and candidates 2+ receive no recent-keyword exemption.

        The core exemption applies only to the underlying taste signal. It never
        permits an exact previously published row recipe or visible row title to
        repeat. Exact row/title identity rotates across all recent generations.
        """
        if history is None:
            history = await self._get_v2_rotation_history(
                token,
                content_type,
            )

        recent_keyword_ids = (
            self._recent_v2_keyword_ids(
                history
            )
        )

        core_window = (
            self._select_v2_prompt_keyword_window(
                features,
                history,
                limit=1,
            )
        )

        core_keyword_id = (
            int(core_window[0][0])
            if core_window
            else None
        )

        @staticmethod
        def normalize_theme_text(
            value: Any,
        ) -> str:
            normalized = "".join(
                char.casefold()
                if char.isalnum()
                else " "
                for char in str(value or "")
            )

            return " ".join(
                normalized.split()
            )

        #
        # Published V2 history records the final visible title and the final
        # build_row_id recipe. Legacy bootstrap rows predate the V2 writer and
        # therefore may have titles without IDs; title protection still applies.
        #
        recent_row_ids: set[str] = set()
        recent_title_keys: set[str] = set()

        for generation in history or []:
            if not isinstance(
                generation,
                dict,
            ):
                continue

            previous_rows = (
                generation.get("rows")
                or []
            )

            if not isinstance(
                previous_rows,
                list,
            ):
                continue

            for previous_row in previous_rows:
                if not isinstance(
                    previous_row,
                    dict,
                ):
                    continue

                previous_id = str(
                    previous_row.get("id")
                    or ""
                )

                if previous_id:
                    recent_row_ids.add(
                        previous_id
                    )

                previous_title = (
                    normalize_theme_text(
                        previous_row.get("title")
                    )
                )

                if previous_title:
                    recent_title_keys.add(
                        previous_title
                    )

        def recent_identity_conflict(
            candidate_row: RowDefinition,
        ) -> str | None:
            """Return the cross-generation duplicate reason, if any."""
            candidate_id = str(
                candidate_row.id
                or ""
            )

            if (
                candidate_id
                and candidate_id
                in recent_row_ids
            ):
                return "recipe"

            title_key = (
                normalize_theme_text(
                    candidate_row.title
                )
            )

            if (
                title_key
                and title_key
                in recent_title_keys
            ):
                return "title"

            return None

        def blocked_recent_themes(
            candidate_index: int,
            candidate_row: RowDefinition,
        ) -> set[int]:
            """Return cooled-down themes present in axes OR visible title."""
            blocked: set[int] = set()

            #
            # First enforce the authoritative resolved TMDB keyword IDs.
            #
            for axis in candidate_row.axes:
                if axis.name != AXIS_KEYWORD:
                    continue

                try:
                    keyword_id = int(
                        axis.value
                    )
                except (TypeError, ValueError):
                    continue

                if keyword_id in recent_keyword_ids:
                    blocked.add(
                        keyword_id
                    )

            #
            # Also prevent a model-generated title from visually recreating a
            # cooled-down theme while using different/fresh query axes.
            #
            normalized_title = normalize_theme_text(
                candidate_row.title
            )

            padded_title = (
                f" {normalized_title} "
            )

            for keyword_id in recent_keyword_ids:
                keyword_name = (
                    features.get_keyword_name(
                        keyword_id
                    )
                )

                normalized_keyword = (
                    normalize_theme_text(
                        keyword_name
                    )
                )

                if not normalized_keyword:
                    continue

                if (
                    f" {normalized_keyword} "
                    in padded_title
                ):
                    blocked.add(
                        keyword_id
                    )

            #
            # Candidate 1 alone may repeat the one intentional stable core signal.
            #
            if (
                candidate_index == 1
                and core_keyword_id is not None
            ):
                blocked.discard(
                    core_keyword_id
                )

            return blocked

        watched_tmdb = (
            await self._get_v2_normalized_watched_tmdb(
                token,
                content_type,
            )
        )

        if watched_tmdb is None:
            logger.warning(
                f"[ThemeRotationV2] watched history unavailable for "
                f"{content_type}; refusing to publish an unvalidated V2 set"
            )
            return None

        accepted: list[RowDefinition] = []

        seen_row_ids: set[str] = set()
        seen_titles: set[str] = set()

        for candidate_index, row in enumerate(
            rows,
            start=1,
        ):
            history_duplicate_before = (
                recent_identity_conflict(
                    row
                )
            )

            if history_duplicate_before:
                logger.info(
                    f"[ThemeRotationV2] candidate {candidate_index} "
                    f"'{row.title}' rejected as recent cross-generation "
                    f"duplicate {history_duplicate_before}: {row.id}"
                )

                continue

            blocked_before = (
                blocked_recent_themes(
                    candidate_index,
                    row,
                )
            )

            if blocked_before:
                blocked_names = [
                    (
                        features.get_keyword_name(
                            keyword_id
                        )
                        or str(keyword_id)
                    )
                    for keyword_id
                    in sorted(blocked_before)
                ]

                logger.info(
                    f"[ThemeRotationV2] candidate {candidate_index} "
                    f"'{row.title}' rejected by final-axis cooldown: "
                    f"{blocked_names}"
                )

                continue

            result = (
                await self._v2_repair_row_for_unseen(
                    row,
                    content_type,
                    watched_tmdb,
                )
            )

            status = str(
                result.get("status")
                or "unknown"
            )

            original_probe = (
                result.get("original_probe")
                or {}
            )

            final_probe = (
                result.get("probe")
                or {}
            )

            final_row = result.get("row")

            logger.info(
                f"[ThemeRotationV2] candidate {candidate_index} "
                f"'{row.title}': status={status} "
                f"original_unseen={original_probe.get('unseen_count')} "
                f"final_unseen={final_probe.get('unseen_count')} "
                f"repair={result.get('repair_reason')}"
            )

            if final_row is None:
                continue

            #
            # Repair can change the recipe, so retitle it before final validation.
            # This ensures the visible title itself is covered by cooldown and
            # duplicate checks rather than being changed after acceptance.
            #
            if result.get("changed"):
                await self._retitle_repaired_rows(
                    [final_row],
                    features,
                )

            #
            # Anthology is a viewer-facing format promise. Validate it
            # deterministically for EVERY final row, including healthy rows
            # that did not otherwise need repair/retitling.
            #
            anthology_required = (
                self._row_has_anthology_keyword(
                    final_row,
                    features,
                )
            )

            anthology_visible = (
                self._title_mentions_anthology(
                    final_row.title
                )
            )

            if anthology_required != anthology_visible:
                logger.warning(
                    f"[AnthologyTitle] candidate {candidate_index} "
                    f"'{final_row.title}' has title/filter mismatch: "
                    f"required={anthology_required} "
                    f"visible={anthology_visible}; retitling"
                )

                await self._retitle_repaired_rows(
                    [final_row],
                    features,
                )

                anthology_visible = (
                    self._title_mentions_anthology(
                        final_row.title
                    )
                )

                if (
                    anthology_required
                    != anthology_visible
                ):
                    logger.warning(
                        f"[AnthologyTitle] candidate "
                        f"{candidate_index} still has "
                        f"title/filter mismatch after retitle: "
                        f"'{final_row.title}'; rejecting "
                        "candidate rather than publishing "
                        "misleading anthology wording"
                    )
                    continue

                logger.info(
                    f"[AnthologyTitle] candidate "
                    f"{candidate_index} corrected to "
                    f"'{final_row.title}'"
                )

            #
            # Repair and retitling can both change cross-generation identity.
            # Validate the ACTUAL final row before cooldown and acceptance.
            #
            history_duplicate_after = (
                recent_identity_conflict(
                    final_row
                )
            )

            if history_duplicate_after:
                logger.warning(
                    f"[ThemeRotationV2] candidate {candidate_index} "
                    f"'{final_row.title}' rejected after repair/retitle "
                    f"as recent cross-generation duplicate "
                    f"{history_duplicate_after}: {final_row.id}"
                )

                continue

            # Enforce cooldown again against the actual final recipe AND title.
            blocked_after = (
                blocked_recent_themes(
                    candidate_index,
                    final_row,
                )
            )

            if blocked_after:
                blocked_names = [
                    (
                        features.get_keyword_name(
                            keyword_id
                        )
                        or str(keyword_id)
                    )
                    for keyword_id
                    in sorted(blocked_after)
                ]

                logger.warning(
                    f"[ThemeRotationV2] candidate {candidate_index} "
                    f"'{final_row.title}' rejected after repair by "
                    f"final-axis cooldown: {blocked_names}"
                )

                continue

            row_id = str(
                final_row.id
            )

            title_key = (
                normalize_theme_text(
                    final_row.title
                )
            )

            if row_id in seen_row_ids:
                logger.info(
                    f"[ThemeRotationV2] skipping duplicate row recipe "
                    f"from candidate {candidate_index}: {row_id}"
                )
                continue

            if (
                title_key
                and title_key in seen_titles
            ):
                logger.info(
                    f"[ThemeRotationV2] skipping duplicate row title "
                    f"from candidate {candidate_index}: "
                    f"'{final_row.title}'"
                )
                continue

            seen_row_ids.add(
                row_id
            )

            if title_key:
                seen_titles.add(
                    title_key
                )

            accepted.append(
                final_row
            )

            if (
                len(accepted)
                >= THEME_ROTATION_V2_PUBLISH_COUNT
            ):
                break

        if (
            len(accepted)
            < THEME_ROTATION_V2_PUBLISH_COUNT
        ):
            logger.warning(
                f"[ThemeRotationV2] only {len(accepted)} of "
                f"{THEME_ROTATION_V2_PUBLISH_COUNT} required "
                f"{content_type} rows survived unseen validation; "
                "discarding this generation"
            )
            return None

        accepted = accepted[
            :THEME_ROTATION_V2_PUBLISH_COUNT
        ]

        logger.info(
            f"[ThemeRotationV2] accepted exactly "
            f"{len(accepted)} publishable {content_type} rows"
        )

        return accepted

    async def _get_v2_rotation_history(
        self,
        token: str | None,
        content_type: str,
    ) -> list[dict[str, Any]]:
        """Return V2 history, bootstrapping from current legacy rows once.

        A brand-new V2 namespace has no memory of the rows the user is already
        seeing under the legacy algorithm. Without this bootstrap, the first V2
        generation can immediately recycle those same themes.

        Once V2 has successful history of its own, that history takes precedence
        and the legacy row cache is no longer consulted.
        """
        if not token:
            return []

        try:
            from app.services.user_cache import user_cache

            history = await user_cache.get_theme_rotation_history(
                token,
                content_type,
            )

            if history:
                return history

        except Exception as exc:
            logger.warning(
                f"[ThemeRotationV2] failed to read V2 history for "
                f"{content_type}: {exc}"
            )

        # Generation-zero bootstrap only. Read the legacy cache directly rather
        # than through _llm_rows_key(), because V2 deliberately uses a separate
        # cache namespace.
        try:
            from app.services.redis_service import redis_service

            legacy_key = (
                f"watchly:llm_rows:{token}:{content_type}"
            )

            raw = await redis_service.get(
                legacy_key
            )

            if not raw:
                return []

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            data = json.loads(raw)

            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = data.get("rows") or []
            else:
                return []

            bootstrap_rows = []

            for row in rows:
                if not isinstance(row, dict):
                    continue

                keyword_ids = []

                for axis in row.get("axes") or []:
                    if not isinstance(axis, dict):
                        continue

                    if axis.get("name") != AXIS_KEYWORD:
                        continue

                    try:
                        keyword_id = int(
                            axis.get("value")
                        )
                    except (TypeError, ValueError):
                        continue

                    keyword_ids.append(
                        keyword_id
                    )

                if keyword_ids:
                    bootstrap_rows.append({
                        "title": row.get("title"),
                        "keyword_ids": keyword_ids,
                    })

            if not bootstrap_rows:
                return []

            logger.info(
                f"[ThemeRotationV2] bootstrapping {content_type} cooldown "
                f"from {len(bootstrap_rows)} current legacy rows"
            )

            return [{
                "source": "legacy_bootstrap",
                "rows": bootstrap_rows,
            }]

        except Exception as exc:
            logger.warning(
                f"[ThemeRotationV2] failed to bootstrap legacy "
                f"{content_type} rows: {exc}"
            )
            return []

    @staticmethod
    def _recent_v2_keyword_ids(
        history: list[dict[str, Any]],
    ) -> set[int]:
        """Collect keyword axes used by recent successful V2 generations.

        The reader accepts both a generation-level ``keyword_ids`` summary
        and per-row ``keyword_ids`` so the stored history can remain useful
        if its representation becomes more detailed later.
        """
        recent: set[int] = set()

        def add_ids(values: Any) -> None:
            if not isinstance(values, (list, tuple, set)):
                return

            for value in values:
                try:
                    recent.add(int(value))
                except (TypeError, ValueError):
                    continue

        for generation in history or []:
            if not isinstance(generation, dict):
                continue

            add_ids(generation.get("keyword_ids"))

            rows = generation.get("rows")
            if not isinstance(rows, list):
                continue

            for row in rows:
                if isinstance(row, dict):
                    add_ids(row.get("keyword_ids"))

        return recent

    @classmethod
    def _select_v2_prompt_keyword_window(
        cls,
        features: "ExtractedFeatures",
        history: list[dict[str, Any]],
        limit: int = THEME_ROTATION_V2_PROMPT_KEYWORD_LIMIT,
    ) -> list[tuple[int, float]]:
        """Choose one familiar signal plus fresh rotating profile keywords.

        Slot 1 intentionally keeps the user's strongest usable profile keyword
        available for the strongest-match row.

        The remaining slots prefer ranked profile keywords that have not appeared
        in the recent successful V2 generations. Recent signals are only reused
        when the deeper profile pool cannot fill the requested window.
        """
        if limit <= 0:
            return []

        pool: list[tuple[int, float]] = []
        seen_ids: set[int] = set()

        for keyword_id, score in features.keywords:
            try:
                keyword_id = int(keyword_id)
            except (TypeError, ValueError):
                continue

            if keyword_id in seen_ids:
                continue

            # A keyword without a resolved name cannot be useful in the LLM hint.
            if not features.get_keyword_name(keyword_id):
                continue

            seen_ids.add(keyword_id)
            pool.append((keyword_id, score))

        if not pool:
            return []

        recent_ids = cls._recent_v2_keyword_ids(history)

        # Keep one strong familiar signal even if it was used recently.
        selected = [pool[0]]

        if limit == 1:
            return selected

        fresh = [
            item
            for item in pool[1:]
            if item[0] not in recent_ids
        ]

        selected.extend(
            fresh[: max(0, limit - len(selected))]
        )

        if len(selected) < limit:
            selected_ids = {
                keyword_id
                for keyword_id, _ in selected
            }

            fallback = [
                item
                for item in pool[1:]
                if item[0] not in selected_ids
            ]

            selected.extend(
                fallback[: max(0, limit - len(selected))]
            )

        return selected[:limit]

    async def _regenerate_in_background(
        self,
        profile: TasteProfile,
        features: "ExtractedFeatures",
        content_type: str,
        api_key: str,
        token: str | None,
    ) -> None:
        """Refresh a stale row set without blocking the request that noticed it."""
        try:
            rows = await self._generate_rows_with_llm(
                profile,
                features,
                content_type,
                api_key,
                token=token,
            )
            if rows:
                if not settings.THEME_ROTATION_V2_ENABLED:
                    # Preserve the existing legacy background behavior exactly.
                    rows = await self._repair_thin_rows(
                        rows,
                        features,
                        content_type,
                    )

                await self._cache_llm_rows(
                    token,
                    content_type,
                    rows,
                )

                logger.info(
                    f"[LLM Rows] background refresh stored {len(rows)} rows for {content_type}"
                )
        except Exception as e:
            logger.warning(f"[LLM Rows] background refresh failed for {content_type}: {e}")

    async def _generate_rows_with_llm(
        self,
        profile: TasteProfile,
        features: ExtractedFeatures,
        content_type: str,
        api_key: str,
        avoid_titles: set[str] | None = None,
        token: str | None = None,
    ) -> list[RowDefinition] | None:
        """Generate rows from the user's interest summary; balance personalization with discovery."""
        try:
            summary = profile.interest_summary or "No summary available."

            current_genre_map = movie_genres if content_type == "movie" else series_genres
            valid_genre_list = ", ".join([f"{name} (ID: {gid})" for gid, name in current_genre_map.items()])

            if settings.THEME_ROTATION_V2_ENABLED:
                history = await self._get_v2_rotation_history(
                    token,
                    content_type,
                )

                keyword_window = self._select_v2_prompt_keyword_window(
                    features,
                    history,
                )

                keyword_names = [
                    features.get_keyword_name(keyword_id)
                    for keyword_id, _ in keyword_window
                ]
                keyword_names = [
                    name
                    for name in keyword_names
                    if name
                ]

                core_keyword = (
                    keyword_names[0]
                    if keyword_names
                    else None
                )
                rotating_keywords = keyword_names[1:]

                keyword_hint_parts = []

                if core_keyword:
                    keyword_hint_parts.append(
                        "Strong familiar profile theme for the strongest-match "
                        f"row: {core_keyword}. "
                        "Do not let this one familiar theme dominate the other rows."
                    )

                if rotating_keywords:
                    keyword_hint_parts.append(
                        "Fresh rotation themes from the user's actual profile; "
                        "prioritize these across the variety, discovery, "
                        "lesser-known and mood rows: "
                        f"{', '.join(rotating_keywords)}."
                    )

                keyword_hint_parts.append(
                    "You can also suggest adjacent discovery themes they would "
                    "likely enjoy. We will resolve keywords."
                )

                keyword_hint = " ".join(keyword_hint_parts)

                recent_ids = self._recent_v2_keyword_ids(
                    history
                )

                recent_keyword_names = [
                    features.get_keyword_name(
                        keyword_id
                    )
                    for keyword_id in sorted(
                        recent_ids
                    )
                ]

                recent_keyword_names = [
                    name
                    for name in recent_keyword_names
                    if name
                ]

                recent_row_titles = []

                for generation in history:
                    for previous_row in (
                        generation.get("rows")
                        or []
                    ):
                        previous_title = (
                            previous_row.get("title")
                            if isinstance(
                                previous_row,
                                dict,
                            )
                            else None
                        )

                        if previous_title:
                            recent_row_titles.append(
                                str(previous_title)
                            )

                if recent_keyword_names:
                    keyword_hint += (
                        " RECENT COOLDOWN: do not recreate these recently used "
                        "themes or use them in titles or keyword choices: "
                        f"{', '.join(recent_keyword_names)}. "
                        "The single strongest-match core row may use its explicitly "
                        "provided core theme when appropriate; other rows must rotate."
                    )

                if recent_row_titles:
                    keyword_hint += (
                        " Recent row concepts/titles that should not be recreated: "
                        f"{', '.join(recent_row_titles)}."
                    )

                logger.info(
                    f"[ThemeRotationV2] {content_type} keyword window: "
                    f"core={core_keyword!r} "
                    f"rotating={rotating_keywords} "
                    f"recent_keyword_count={len(recent_ids)}"
                )

            else:
                # Legacy prompt construction must remain exactly unchanged.
                profile_keywords = [
                    name
                    for k_id, _ in features.keywords[:12]
                    if (name := features.get_keyword_name(k_id))
                ]
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
                f" {content_type}. Use genres (required) and keywords. Production country must not influence a row.\n\nInterest"
                f" Summary:\n{summary}\n\nGenerate 5 rows. The bracketed labels are internal planning labels only and must NEVER appear in a title:\n1. [strongest match] — What they will love"
                " most: strongest match to their taste (genres + keywords).\n2. [variety]"
                " — Blend of their tastes with more variety (genres + keywords).\n3. [discovery] — Discovery: suggest themes they might not have explored yet but"
                " would likely enjoy (adjacent to their taste, or natural next step). Use genres + keywords;"
                " openness to new content here.\n4. [lesser-known] — Lesser-known or cult titles matching"
                " their taste. Use 1-2 genres + 1 keyword max.\n5. [mood] — A specific mood or tone they"
                " would enjoy (e.g. mind-bending, atmospheric, tense). Use genres + 1 keyword max.\n\nRules:\n"
                "- Genres: use ONLY these TMDB Genre IDs:"
                f" {valid_genre_list}\n- Keywords: {keyword_hint}\n- Country: ALWAYS null. Production country is not a user taste signal and must never define or name a row."
                "\n- TITLE RULE (most important): the title must describe WHAT THE FILMS ARE, never the row's purpose. Never use these words: core, mixed, rising, deep cut, mood, picks, favorites, selection, collection, essentials, hits, vibes. Name the most DISTINCTIVE genre or thematic constraint rather than summarising every axis. Good: Crime+Drama, 'based on novel or book' -> Literary Crime. Sci-Fi+Thriller, 'artificial intelligence' -> Rogue AI Thrillers. Documentary, 'true crime' -> True Crime Investigations. Bad: Core Favorites, Mixed Dramas, Rising Mysteries, Deep Cuts, Mood Picks.\n- Each row: title (2-5 words), genres (list of IDs), keywords (list"
                " of strings), country (string or null).\n- IMPORTANT: Keep combinations simple and achievable."
                " Use max 2 genres and max 1-2 keywords per row. Do NOT combine 3+ niche constraints together"
                " (e.g. avoid Documentary + dark comedy + anthology — too niche).\n- Output a JSON array of 5 objects."
                + avoid_clause
            )

            if settings.THEME_ROTATION_V2_ENABLED:
                prompt = prompt.replace(
                    "- TITLE RULE (most important): the title must describe WHAT THE FILMS ARE, never the row's purpose. Never use these words: core, mixed, rising, deep cut, mood, picks, favorites, selection, collection, essentials, hits, vibes. Name the most DISTINCTIVE genre or thematic constraint rather than summarising every axis. Good: Crime+Drama, 'based on novel or book' -> Literary Crime. Sci-Fi+Thriller, 'artificial intelligence' -> Rogue AI Thrillers. Documentary, 'true crime' -> True Crime Investigations. Bad: Core Favorites, Mixed Dramas, Rising Mysteries, Deep Cuts, Mood Picks.",
                    (
                        "- TITLE RULE (most important): "
                        + THEME_ROTATION_V2_TITLE_STYLE_GUIDANCE
                    ),
                    1,
                )

                prompt = prompt.replace(
                    "generate exactly 5 streaming collections",
                    (
                        "generate exactly "
                        f"{THEME_ROTATION_V2_CANDIDATE_COUNT} "
                        "candidate streaming collections"
                    ),
                    1,
                )

                prompt = prompt.replace(
                    "Generate 5 rows.",
                    (
                        "Generate "
                        f"{THEME_ROTATION_V2_CANDIDATE_COUNT} "
                        "candidate rows. Only five will ultimately be published "
                        "after inventory validation."
                    ),
                    1,
                )

                prompt = prompt.replace(
                    (
                        "5. [mood] — A specific mood or tone they"
                        " would enjoy (e.g. mind-bending, atmospheric, tense). "
                        "Use genres + 1 keyword max.\n\nRules:\n"
                    ),
                    (
                        "5. [mood] — A specific mood or tone they"
                        " would enjoy (e.g. mind-bending, atmospheric, tense). "
                        "Use genres + 1 keyword max.\n"
                        + "".join(
                            (
                                f"{candidate_index}. [alternate] — "
                                "A genuinely different but still taste-grounded "
                                "alternate theme using a different rotation signal; "
                                "avoid near-duplicates of earlier candidates.\n"
                            )
                            for candidate_index in range(
                                6,
                                THEME_ROTATION_V2_CANDIDATE_COUNT + 1,
                            )
                        )
                        + "\nRules:\n"
                    ),
                    1,
                )

                prompt = prompt.replace(
                    "Output a JSON array of 5 objects.",
                    (
                        "Output a JSON array of exactly "
                        f"{THEME_ROTATION_V2_CANDIDATE_COUNT} objects."
                    ),
                    1,
                )

                system_instruction = (
                    "You are a creative streaming curator. Design "
                    f"{THEME_ROTATION_V2_CANDIDATE_COUNT} candidate catalog rows "
                    "from the user's interest summary. The first five follow the "
                    "requested core/variety/discovery/lesser-known/mood plan. "
                    "The remaining candidates are distinct alternates so inventory "
                    "validation can reject a thin theme without losing a row. "
                    "Use genres and keywords only. Country must always be null. "
                    + THEME_ROTATION_V2_TITLE_STYLE_GUIDANCE
                    + " Output valid JSON only."
                )
            else:
                # Exact legacy system instruction.
                system_instruction = (
                    "You are a creative film curator. Design 5 catalog rows from the user's interest summary."
                    " Row 1 (The Core): strong match. Row 2 (Mixed): blend + variety. Row 3 (Rising Star):"
                    " discovery—suggest new content they would enjoy, not just more of the same. Use genres"
                    " and keywords only. Country must always be null. Output valid JSON only."
                )

            data = await gemini_service.generate_structured_async(
                prompt=prompt,
                response_schema=list[LLMRowTheme],
                system_instruction=system_instruction,
                api_key=api_key,
                google_api_key=(
                    (
                        getattr(
                            self.user_settings,
                            "gemini_api_key",
                            None,
                        )
                        if self.user_settings
                        else None
                    )
                    or settings.GEMINI_API_KEY
                ),
            )

            if isinstance(data, list):
                logger.info(
                    f"[LLM Rows] model returned {len(data)} raw themes for {content_type}: "
                    + "; ".join(
                        f"{d.get('title', '?')}"
                        f"[g={d.get('genres')} k={d.get('keywords')} c={d.get('country')}]"
                        for d in data
                        if isinstance(d, dict)
                    )
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

            # Track anchor genres across rows to encourage variety. Keywords may repeat
            # when paired with different genres; suppressing reuse erased valid row themes.
            used_anchor_genres: set[int] = set()

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

                # Enforce the prompt's hard complexity limits even when the model
                # returns more. Extra genres were creating needlessly tiny AND pools.
                genre_ids = list(genre_ids or [])[:2]
                kw_names = list(kw_names or [])[:2]

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

                # If every genre this row asked for was already anchored by an earlier
                # row, the builder ends up with no anchor at all and build() returns
                # None -- silently discarding the row. With a concentrated taste
                # profile (sci-fi/crime/thriller) this killed 2 of every 5 themes.
                # An overlapping discover pool is much better than a missing row:
                # keywords and country still differentiate them, and AND semantics
                # means the resulting queries are not actually the same.
                if not row_anchor_genres and genre_ids:
                    for gid in genre_ids:
                        gid = int(gid)
                        if gid in current_genre_map:
                            # The genre is already present as FLAVOR when it was
                            # anchored by an earlier row. Promote that existing axis
                            # instead of appending a duplicate ANCHOR copy.
                            existing_axis = next(
                                (
                                    a
                                    for a in builder.components.axes
                                    if a.name == AXIS_GENRE and int(a.value) == gid
                                ),
                                None,
                            )
                            if existing_axis is not None:
                                existing_axis.role = AxisRole.ANCHOR
                            else:
                                builder.add_axis(AXIS_GENRE, gid, AxisRole.ANCHOR)
                            row_anchor_genres.append(gid)
                            logger.debug(
                                f"[LLM Rows] '{title}' had no free anchor genre; "
                                f"re-anchoring on {gid} to keep the row"
                            )
                            break

                for kw_name in kw_names:
                    kid = await self._resolve_keyword_to_id(kw_name, profile_kw_map)
                    if kid is not None and kid not in GENERIC_KEYWORD_BLACKLIST:
                        # Keywords discovered outside the cached profile still need a
                        # display name so repaired titles can be grounded accurately.
                        features.keyword_names.setdefault(kid, str(kw_name).strip())
                        builder.add_axis(AXIS_KEYWORD, kid, AxisRole.FLAVOR)

                # Ignore country even if a model violates the prompt and returns one.
                row_comp = builder.build()
                if row_comp and row_comp.axes:
                    row_id = build_row_id(row_comp.axes)
                    final_rows.append(RowDefinition(title=title, id=row_id, axes=row_comp.axes))
                    used_anchor_genres.update(row_anchor_genres)

            logger.info(
                f"[LLM Rows] {len(final_rows)} of {len(data)} themes survived processing "
                f"for {content_type}"
            )
            if final_rows:
                if settings.THEME_ROTATION_V2_ENABLED:
                    return await self._select_v2_publishable_rows(
                        final_rows,
                        features,
                        content_type,
                        token,
                        history=history,
                    )

                # Exact legacy raw-inventory repair path.
                final_rows = await self._repair_thin_rows(
                    final_rows,
                    features,
                    content_type,
                )

            return final_rows if final_rows else None

        except Exception as e:
            logger.warning(f"Error in _generate_rows_with_llm: {e}")
            return None
