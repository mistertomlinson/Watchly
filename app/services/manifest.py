import asyncio
from typing import Any

from fastapi import HTTPException
from loguru import logger

from app.core.config import settings
from app.core.security import redact_token
from app.core.settings import UserSettings, resolve_tmdb_api_key
from app.core.version import __version__
from app.services.catalog import DynamicCatalogService
from app.services.profile.integration import ProfileIntegration
from app.services.stremio.service import StremioBundle
from app.services.token_store import token_store
from app.services.translation import apply_catalog_translation
from app.services.user_cache import user_cache
from app.utils.catalog import cache_profile_and_watched_sets, sort_catalogs


class ManifestService:
    """Service for generating Stremio manifest files."""

    def __init__(self):
        self._manifest_prepare_tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def get_base_manifest() -> dict[str, Any]:
        return {
            "id": settings.ADDON_ID,
            "version": __version__,
            "name": settings.ADDON_NAME,
            "description": "Movie and series recommendations based on your connected library.",
            "logo": "https://raw.githubusercontent.com/TimilsinaBimal/Watchly/refs/heads/main/app/static/logo.png",
            "background": "https://raw.githubusercontent.com/TimilsinaBimal/Watchly/refs/heads/main/app/static/cover.png",
            "resources": ["catalog"],
            "types": ["movie", "series"],
            "idPrefixes": ["tt"],
            "catalogs": [],
            "behaviorHints": {"configurable": True, "configurationRequired": False},
            "stremioAddonsConfig": {
                "issuer": "https://stremio-addons.net",
                "signature": (
                    "eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..WSrhzzlj1TuDycD6QoVLuA.Dzmxzr4y83uqQF15r4tC1bB9-vtZRh1Rvy4BqgDYxu91c2esiJuov9KnnI_cboQCgZS7hjwnIqRSlQ-jEyGwXHHRerh9QklyfdxpXqNUyBgTWFzDOVdVvDYJeM_tGMmR.sezAChlWGV7lNS-t9HWB6A"
                ),
            },
        }

    async def _resolve_auth_key(self, bundle: StremioBundle, credentials: dict[str, Any], token: str) -> str | None:
        auth_key = credentials.get("authKey")
        email = credentials.get("email")
        password = credentials.get("password")

        is_valid = False
        if auth_key:
            try:
                await bundle.auth.get_user_info(auth_key)
                is_valid = True
            except Exception as exc:
                logger.debug(f"Auth key check failed for {email or 'unknown'}: {exc}")

        if not is_valid and email and password:
            try:
                auth_key = await bundle.auth.login(email, password)
                credentials["authKey"] = auth_key
                await token_store.update_user_data(token, credentials)
            except Exception as exc:
                logger.error(f"Failed to refresh auth key during manifest fetch: {exc}")
                return None
        return auth_key

    async def cache_library_and_profiles(
        self, bundle: StremioBundle, auth_key: str, user_settings: UserSettings, token: str
    ) -> dict[str, Any]:
        logger.info(f"[{redact_token(token)}] Fetching library items for caching")
        library_items = await bundle.library.get_library_items(auth_key)
        await user_cache.set_library_items(token, library_items)

        integration_service = ProfileIntegration(
            language=user_settings.language,
            tmdb_api_key=resolve_tmdb_api_key(user_settings),
        )
        for content_type in ["movie", "series"]:
            try:
                await cache_profile_and_watched_sets(
                    token, content_type, integration_service, library_items, bundle, auth_key
                )
            except Exception as exc:
                logger.warning(f"[{redact_token(token)}] Failed to build/cache profile for {content_type}: {exc}")
        return library_items

    async def cache_library_and_profiles_from_items(
        self, library_items: dict, user_settings: UserSettings, token: str
    ) -> None:
        await user_cache.set_library_items(token, library_items)
        integration_service = ProfileIntegration(
            language=user_settings.language,
            tmdb_api_key=resolve_tmdb_api_key(user_settings),
        )
        for content_type in ["movie", "series"]:
            try:
                profile, watched_tmdb, watched_imdb = await integration_service.build_profile_from_library(
                    library_items, content_type
                )
                await user_cache.set_profile_and_watched_sets(
                    token, content_type, profile, watched_tmdb, watched_imdb
                )
            except Exception as exc:
                logger.warning(f"[{redact_token(token)}] Failed to cache profile for {content_type}: {exc}")

    async def _ensure_library_and_profiles_cached(
        self, bundle: StremioBundle, auth_key: str, user_settings: UserSettings, token: str
    ) -> dict[str, Any]:
        library_items = await user_cache.get_library_items(token)
        if library_items:
            return library_items
        return await self.cache_library_and_profiles(bundle, auth_key, user_settings, token)

    async def _build_dynamic_catalogs(
        self, bundle: StremioBundle, auth_key: str, user_settings: UserSettings, token: str
    ) -> list[dict[str, Any]]:
        library_items = await user_cache.get_library_items(token)
        if not library_items:
            library_items = await self._ensure_library_and_profiles_cached(bundle, auth_key, user_settings, token)
        service = DynamicCatalogService(
            language=user_settings.language,
            tmdb_api_key=resolve_tmdb_api_key(user_settings),
        )
        return await service.get_dynamic_catalogs(library_items, user_settings, token=token)

    async def _fetch_provider_library(self, provider: str, creds: dict[str, Any]) -> dict[str, Any]:
        access_token = creds.get("authKey")
        if not access_token:
            raise RuntimeError(f"Missing {provider} access token")

        if provider == "trakt":
            from app.services.trakt.service import TraktBundle

            if not settings.TRAKT_CLIENT_ID or not settings.TRAKT_CLIENT_SECRET:
                raise RuntimeError("Trakt server credentials are not configured")
            bundle = TraktBundle(
                client_id=settings.TRAKT_CLIENT_ID,
                client_secret=settings.TRAKT_CLIENT_SECRET,
                redirect_uri=f"{settings.HOST_NAME}/tokens/trakt/callback",
                access_token=access_token,
            )
            try:
                return await bundle.library.get_library_items()
            finally:
                await bundle.close()

        if provider == "simkl":
            from app.services.simkl_provider import SimklApiClient, SimklLibraryProvider

            if not settings.SIMKL_CLIENT_ID:
                raise RuntimeError("Simkl server credentials are not configured")
            client = SimklApiClient(settings.SIMKL_CLIENT_ID, access_token)
            try:
                return await SimklLibraryProvider(client).get_library_items()
            finally:
                await client.close()

        raise RuntimeError(f"Unsupported library provider: {provider}")

    async def _build_dynamic_catalogs_provider(
        self,
        provider: str,
        creds: dict[str, Any],
        user_settings: UserSettings,
        token: str,
    ) -> list[dict[str, Any]]:
        library_items = await user_cache.get_library_items(token)
        if not library_items:
            logger.info(f"[{redact_token(token)}] {provider} library cache expired; refreshing")
            library_items = await self._fetch_provider_library(provider, creds)
            await self.cache_library_and_profiles_from_items(library_items, user_settings, token)

        if not library_items:
            return []

        service = DynamicCatalogService(
            language=user_settings.language,
            tmdb_api_key=resolve_tmdb_api_key(user_settings),
        )
        return await service.get_dynamic_catalogs(library_items, user_settings, token=token)

    async def _translate_catalogs(self, catalogs: list[dict[str, Any]], language: str | None) -> list[dict[str, Any]]:
        if not language or language.startswith("en"):
            return catalogs
        translated = []
        for catalog in catalogs:
            await apply_catalog_translation(catalog, language)
            translated.append(catalog)
        return translated

    def _sort_catalogs(
        self, catalogs: list[dict[str, Any]], user_settings: UserSettings | None
    ) -> list[dict[str, Any]]:
        return sort_catalogs(catalogs, user_settings) if user_settings else catalogs

    async def _build_candidate_manifest_for_token(self, token: str) -> dict[str, Any]:
        creds = await token_store.get_user_data(token)
        if not creds:
            raise HTTPException(status_code=401, detail="Token not found. Please reconfigure the addon.")

        user_settings = UserSettings(**creds.get("settings", {}))
        base_manifest = self.get_base_manifest()
        provider = creds.get("auth_provider")

        if provider in {"trakt", "simkl"}:
            fetched_catalogs = await self._build_dynamic_catalogs_provider(
                provider, creds, user_settings, token
            )
        else:
            bundle = StremioBundle()
            try:
                auth_key = await self._resolve_auth_key(bundle, creds, token)
                fetched_catalogs = (
                    await self._build_dynamic_catalogs(bundle, auth_key, user_settings, token)
                    if auth_key
                    else []
                )
            finally:
                await bundle.close()

        all_catalogs = [c.copy() for c in base_manifest["catalogs"]] + [
            c.copy() for c in fetched_catalogs
        ]
        translated = await self._translate_catalogs(all_catalogs, user_settings.language)
        sorted_catalogs = self._sort_catalogs(translated, user_settings)
        if sorted_catalogs:
            base_manifest["catalogs"] = sorted_catalogs
        return base_manifest

    async def _prewarm_manifest_catalogs(
        self,
        token: str,
        manifest: dict[str, Any],
    ) -> None:
        # Lazy import avoids manifest -> catalog_service -> catalog_updater -> manifest
        # during module initialization.
        from app.services.recommendation.catalog_service import CatalogService

        builder = CatalogService()
        catalogs = manifest.get("catalogs", [])
        total = len(catalogs)

        for index, catalog in enumerate(catalogs, start=1):
            content_type = catalog.get("type")
            catalog_id = catalog.get("id")
            if not content_type or not catalog_id:
                raise RuntimeError(f"Candidate manifest contained invalid catalog definition: {catalog!r}")

            cached = await user_cache.get_catalog(token, content_type, catalog_id)
            if cached is not None:
                continue

            logger.info(
                f"[{redact_token(token)}...] Prewarming next manifest "
                f"catalog {index}/{total}: {content_type}/{catalog_id}"
            )
            await builder.get_catalog(token, content_type, catalog_id)

            cached = await user_cache.get_catalog(token, content_type, catalog_id)
            if cached is None:
                raise RuntimeError(
                    "Catalog build completed without publishing a cache entry for "
                    f"{content_type}/{catalog_id}"
                )

    @staticmethod
    def _catalog_meta_identity(meta: dict[str, Any]) -> str | None:
        """Return a stable identity for cross-row dedupe without using titles."""
        value = meta.get("id") or meta.get("_tmdb_id")
        if value is None or value == "":
            return None
        return str(value)

    async def _dedupe_prewarmed_catalogs(
        self,
        token: str,
        manifest: dict[str, Any],
    ) -> None:
        """Balance duplicate titles across fully built recommendation rows.

        Movies and series are independent pools. Every title that appears in only
        one row is retained there. For titles shared by multiple rows, assign each
        title to the row with the fewest titles retained so far, recalculating after
        every assignment. If rows are tied, keep the title in the row where it
        ranked higher; manifest order is the final deterministic tie-break.

        This intentionally operates only after every candidate-manifest catalog is
        prewarmed, because a single catalog build cannot know the sizes or overlap
        of its sibling rows.
        """
        catalogs = manifest.get("catalogs", [])

        for content_type in ("movie", "series"):
            rows: list[dict[str, Any]] = []

            for position, catalog in enumerate(catalogs):
                if catalog.get("type") != content_type:
                    continue

                catalog_id = catalog.get("id")
                if not catalog_id:
                    continue

                cached = await user_cache.get_catalog(token, content_type, catalog_id)
                if cached is None:
                    raise RuntimeError(
                        "Cannot dedupe candidate manifest because a prewarmed catalog "
                        f"is missing: {content_type}/{catalog_id}"
                    )

                visible_data, _created_at, source_revision = cached
                raw_cached = await user_cache.get_raw_catalog(token, content_type, catalog_id)
                if raw_cached is not None and raw_cached[2] == source_revision:
                    data = raw_cached[0]
                else:
                    # First run after this feature is deployed, or a raw snapshot
                    # from a different source revision: seed it from the complete
                    # visible build before any coordinated dedupe rewrite occurs.
                    data = visible_data
                    await user_cache.set_raw_catalog(
                        token,
                        content_type,
                        catalog_id,
                        data,
                        ttl=None,
                        source_revision=source_revision,
                    )

                metas = list((data or {}).get("metas") or [])
                ordered_ids: list[str] = []
                rank: dict[str, int] = {}
                unidentified_count = 0

                for index, meta in enumerate(metas):
                    media_id = self._catalog_meta_identity(meta)
                    if media_id is None:
                        unidentified_count += 1
                        continue
                    if media_id in rank:
                        continue
                    rank[media_id] = index
                    ordered_ids.append(media_id)

                rows.append(
                    {
                        "position": position,
                        "catalog_id": catalog_id,
                        "name": catalog.get("name") or catalog_id,
                        "data": data,
                        "metas": metas,
                        "ordered_ids": ordered_ids,
                        "rank": rank,
                        "unidentified_count": unidentified_count,
                        "source_revision": source_revision,
                    }
                )

            owners: dict[str, list[int]] = {}
            for row_index, row in enumerate(rows):
                for media_id in row["ordered_ids"]:
                    owners.setdefault(media_id, []).append(row_index)

            duplicates = {
                media_id: indexes
                for media_id, indexes in owners.items()
                if len(indexes) > 1
            }
            if not duplicates:
                continue

            retained: list[set[str]] = [set() for _ in rows]
            for media_id, indexes in owners.items():
                if len(indexes) == 1:
                    retained[indexes[0]].add(media_id)

            def current_size(row_index: int) -> int:
                return len(retained[row_index]) + rows[row_index]["unidentified_count"]

            # Allocate row-first rather than duplicate-first. Picking a fixed duplicate
            # order can strand a genuinely small row: each of its shared titles may be
            # awarded elsewhere early, even though those other rows later grow larger.
            # Water-filling from the currently smallest row makes "smallest row wins"
            # true across the allocation as a whole, not merely for each title in an
            # arbitrary processing order.
            unassigned = dict(duplicates)

            while unassigned:
                eligible_rows: dict[int, list[str]] = {}
                for media_id, indexes in unassigned.items():
                    for row_index in indexes:
                        eligible_rows.setdefault(row_index, []).append(media_id)

                winner = min(
                    eligible_rows,
                    key=lambda i: (
                        current_size(i),
                        len(rows[i]["ordered_ids"]),
                        rows[i]["position"],
                    ),
                )

                media_id = min(
                    eligible_rows[winner],
                    key=lambda mid: (
                        len(unassigned[mid]),
                        rows[winner]["rank"][mid],
                        mid,
                    ),
                )
                retained[winner].add(media_id)
                unassigned.pop(media_id)

            total_removed = 0
            for row_index, row in enumerate(rows):
                new_metas = []
                seen_local: set[str] = set()

                for meta in row["metas"]:
                    media_id = self._catalog_meta_identity(meta)
                    if media_id is None:
                        new_metas.append(meta)
                        continue
                    if media_id in seen_local:
                        continue
                    seen_local.add(media_id)
                    if media_id in retained[row_index]:
                        new_metas.append(meta)

                removed = len(row["metas"]) - len(new_metas)

                # Always rewrite from the undeduped raw snapshot. A previous
                # balancing pass may have removed titles that this allocation
                # now awards back to the row; skipping a zero-removal row would
                # leave that older, smaller visible cache in place.
                rewritten = dict(row["data"])
                rewritten["metas"] = new_metas
                await user_cache.set_catalog(
                    token,
                    content_type,
                    row["catalog_id"],
                    rewritten,
                    ttl=None,
                    source_revision=row["source_revision"],
                )
                total_removed += max(removed, 0)
                if removed > 0:
                    logger.info(
                        f"[{redact_token(token)}...] [Dedupe] {content_type} "
                        f"'{row['name']}' {len(row['metas'])} -> {len(new_metas)} "
                        f"(removed {removed} duplicate placements)"
                    )

            logger.info(
                f"[{redact_token(token)}...] [Dedupe] {content_type}: "
                f"{len(duplicates)} duplicated titles balanced across {len(rows)} rows; "
                f"removed {total_removed} duplicate placements"
            )

    async def _manifest_catalogs_are_cached(
        self,
        token: str,
        manifest: dict[str, Any],
    ) -> bool:
        for catalog in manifest.get("catalogs", []):
            content_type = catalog.get("type")
            catalog_id = catalog.get("id")
            if not content_type or not catalog_id:
                return False
            if await user_cache.get_catalog(token, content_type, catalog_id) is None:
                return False
        return True

    async def _prepare_next_manifest(self, token: str) -> None:
        try:
            if await user_cache.get_prepared_manifest(token):
                return

            candidate = await self._build_candidate_manifest_for_token(token)
            active = await user_cache.get_active_manifest(token)

            if active and active.get("catalogs") and not candidate.get("catalogs"):
                raise RuntimeError("Candidate manifest unexpectedly contained no catalogs")

            await self._prewarm_manifest_catalogs(token, candidate)
            await self._dedupe_prewarmed_catalogs(token, candidate)
            await user_cache.set_prepared_manifest(token, candidate)
            logger.info(
                f"[{redact_token(token)}...] Next Watchly manifest is fully "
                "prewarmed and waiting for the next manifest request"
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                f"[{redact_token(token)}...] Failed to prepare next Watchly "
                f"manifest; keeping current published generation: {exc}"
            )

    def _finish_manifest_prepare(self, token: str, task: asyncio.Task) -> None:
        if self._manifest_prepare_tasks.get(token) is task:
            self._manifest_prepare_tasks.pop(token, None)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(
                f"[{redact_token(token)}...] Manifest prepare task ended with error: {exc}"
            )

    def _schedule_next_manifest_prepare(self, token: str) -> None:
        existing = self._manifest_prepare_tasks.get(token)
        if existing is not None and not existing.done():
            return

        task = asyncio.create_task(
            self._prepare_next_manifest(token),
            name=f"watchly-manifest-prepare-{redact_token(token)}",
        )
        self._manifest_prepare_tasks[token] = task
        task.add_done_callback(
            lambda completed, user_token=token: self._finish_manifest_prepare(user_token, completed)
        )

    async def get_manifest_for_token(self, token: str) -> dict[str, Any]:
        if not token:
            raise HTTPException(status_code=401, detail="Missing token. Please reconfigure the addon.")

        # A generation prepared invisibly during the previous request/launch is
        # promoted before the ordinary fresh-cache check.
        prepared = await user_cache.get_prepared_manifest(token)
        if prepared:
            if await self._manifest_catalogs_are_cached(token, prepared):
                await user_cache.set_manifest(token, prepared)
                await user_cache.clear_prepared_manifest(token)
                logger.info(f"[{redact_token(token)}] Promoted fully prewarmed Watchly manifest generation")
                self._schedule_next_manifest_prepare(token)
                return prepared

            # A prepared generation may outlive one of its catalog cache entries
            # if the user does not launch for a long time. Never promote it cold.
            logger.warning(
                f"[{redact_token(token)}] Prepared manifest lost one or more "
                "catalog cache entries; discarding it instead of publishing cold IDs"
            )
            await user_cache.clear_prepared_manifest(token)

        cached = await user_cache.get_manifest(token)
        if cached:
            self._schedule_next_manifest_prepare(token)
            return cached

        # If the short manifest TTL expired while the next generation failed or
        # was still building, keep serving the last known-good published snapshot.
        active = await user_cache.get_active_manifest(token)
        if active:
            await user_cache.set_manifest(token, active)
            logger.info(
                f"[{redact_token(token)}] Fresh manifest cache expired; "
                "re-serving last published generation while next one prepares"
            )
            self._schedule_next_manifest_prepare(token)
            return active

        # Bootstrap only: never advertise brand-new catalog IDs before their
        # payloads exist. Prepare the first full generation invisibly.
        creds = await token_store.get_user_data(token)
        if not creds:
            raise HTTPException(status_code=401, detail="Token not found. Please reconfigure the addon.")

        base_manifest = self.get_base_manifest()
        await user_cache.set_manifest(token, base_manifest)
        logger.info(
            f"[{redact_token(token)}] No published manifest snapshot yet; "
            "serving base manifest while first generation prewarms"
        )
        self._schedule_next_manifest_prepare(token)
        return base_manifest


manifest_service = ManifestService()
