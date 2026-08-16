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
