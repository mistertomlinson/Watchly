from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

from loguru import logger

from app.core.constants import DEFAULT_CONCURRENCY_LIMIT

# TMDB release_dates types: 1=Premiere, 2=Theatrical (limited),
# 3=Theatrical, 4=Digital, 5=Physical, 6=TV.
HOME_RELEASE_TYPES = {4, 5}

# TMDB's digital/physical release metadata is much less complete for older
# catalog titles. We only require an explicit home-release record for movies
# that are recent enough to plausibly still be theatrical-only. Older movies
# are treated as available once their theatrical release is at least a year old.
RECENT_THEATRICAL_WINDOW_DAYS = 365


def _parse_tmdb_date(raw: Any) -> date | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip()[:10])
    except ValueError:
        return None


def _preferred_region(user_settings: Any) -> str | None:
    language = str(getattr(user_settings, "language", "") or "").strip()
    if "-" not in language:
        return None
    region = language.rsplit("-", 1)[-1].upper()
    return region if len(region) == 2 and region.isalpha() else None


def _tmdb_id(item: dict[str, Any]) -> int | None:
    raw = item.get("_tmdb_id") or item.get("tmdb_id") or item.get("id")
    if isinstance(raw, str) and raw.startswith("tmdb:"):
        raw = raw.split(":", 1)[1]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _has_home_release(
    payload: dict[str, Any] | None,
    *,
    today: date,
    preferred_region: str | None,
) -> bool:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return False

    # Prefer the user's language region when TMDB has an entry for it. If TMDB
    # has no entry for that region at all, fall back to worldwide data rather
    # than treating a missing country bucket as proof of unavailability.
    scoped_results = results
    if preferred_region:
        regional = [
            country
            for country in results
            if isinstance(country, dict)
            and str(country.get("iso_3166_1") or "").upper() == preferred_region
        ]
        if regional:
            scoped_results = regional

    for country in scoped_results:
        if not isinstance(country, dict):
            continue
        for release in country.get("release_dates", []) or []:
            if not isinstance(release, dict):
                continue
            try:
                release_type = int(release.get("type"))
            except (TypeError, ValueError):
                continue
            if release_type not in HOME_RELEASE_TYPES:
                continue
            release_date = _parse_tmdb_date(release.get("release_date"))
            if release_date is not None and release_date <= today:
                return True

    return False


async def filter_home_released_movies(
    tmdb_service: Any,
    items: list[dict[str, Any]],
    user_settings: Any = None,
    *,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Remove recent/future movies that are not home-viewable yet.

    A recent movie is publishable only when TMDB has a Digital (type 4) or
    Physical/Blu-ray (type 5) release on or before today. This deliberately does
    not accept a theatrical/premiere date as proof that a movie is watchable at
    home. For older titles we fail open after one year because TMDB's historical
    digital/physical release records are incomplete and those titles are no
    longer credible theatrical-only false positives.

    The gate runs on the small, already-ranked Top Picks result set, so it adds
    only a bounded number of TMDB release-date requests and preserves V2's large
    candidate-pool/scoring pipeline.
    """
    if not items:
        return []

    today = today or date.today()
    old_enough_cutoff = today - timedelta(days=RECENT_THEATRICAL_WINDOW_DAYS)
    region = _preferred_region(user_settings)
    semaphore = asyncio.Semaphore(min(DEFAULT_CONCURRENCY_LIMIT, 12))

    async def _check(item: dict[str, Any]) -> tuple[bool, str]:
        theatrical_date = _parse_tmdb_date(item.get("released"))

        if theatrical_date is not None and theatrical_date > today:
            return False, f"theatrical release {theatrical_date.isoformat()} is in the future"

        # Old catalog movies should not be lost because TMDB never backfilled a
        # digital/Blu-ray date for them.
        if theatrical_date is not None and theatrical_date <= old_enough_cutoff:
            return True, "older catalog title"

        tmdb_id = _tmdb_id(item)
        if tmdb_id is None:
            return False, "recent/unknown movie has no TMDB identity for availability verification"

        try:
            async with semaphore:
                payload = await tmdb_service.client.get(f"/movie/{tmdb_id}/release_dates")
        except Exception as exc:
            logger.warning(
                "[HomeAvailability] Could not verify recent movie "
                f"tmdb={tmdb_id} title={item.get('name')!r}: {exc}"
            )
            return False, "home-release lookup failed"

        if _has_home_release(payload, today=today, preferred_region=region):
            return True, "digital/physical release is available"

        return False, "no digital/physical release on or before today"

    checks = await asyncio.gather(*(_check(item) for item in items))

    kept: list[dict[str, Any]] = []
    for item, (allowed, reason) in zip(items, checks):
        if allowed:
            kept.append(item)
            continue

        logger.info(
            "[HomeAvailability] Excluding Top Pick "
            f"tmdb={_tmdb_id(item)} title={item.get('name')!r}: {reason}"
        )

    if len(kept) != len(items):
        logger.info(
            f"[HomeAvailability] Kept {len(kept)}/{len(items)} movie Top Picks "
            "after digital/physical availability gate"
        )

    return kept
