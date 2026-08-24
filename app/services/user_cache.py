import hashlib
import json
import time
from typing import Any

from loguru import logger
from redis.exceptions import WatchError

from app.core.config import settings
from app.core.constants import CATALOG_KEY, LIBRARY_ITEMS_KEY, PROFILE_KEY, WATCHED_SETS_KEY

RAW_CATALOG_KEY = "watchly:catalog_raw:{token}:{type}:{id}"
STAGED_CATALOG_KEY = "watchly:catalog_staged:{token}:{type}:{id}"
STAGED_RAW_CATALOG_KEY = "watchly:catalog_staged_raw:{token}:{type}:{id}"
from app.core.security import redact_token
from app.models.taste_profile import TasteProfile
from app.services.redis_service import redis_service

# Derived caches (library, profiles, watched sets, hashes) are rebuilt on demand
# from the upstream provider, so they do not need to live forever. Without a TTL
# these keys accumulate indefinitely for deleted or abandoned tokens.
DERIVED_CACHE_TTL_SECONDS = 2592000  # 30 days


class UserCacheService:
    @staticmethod
    def _library_items_key(token: str) -> str:
        """Generate cache key for library items."""
        return LIBRARY_ITEMS_KEY.format(token=token)

    @staticmethod
    def _profile_key(token: str, content_type: str) -> str:
        """Generate cache key for profile."""
        return PROFILE_KEY.format(token=token, content_type=content_type)

    @staticmethod
    def _watched_sets_key(token: str, content_type: str) -> str:
        """Generate cache key for watched sets."""
        return WATCHED_SETS_KEY.format(token=token, content_type=content_type)

    @staticmethod
    def _library_hash_key(token: str, content_type: str) -> str:
        """Generate cache key for library hash."""
        return f"watchly:library_hash:{token}:{content_type}"

    @staticmethod
    def _last_profile_build_key(token: str, content_type: str) -> str:
        """Generate cache key for last profile build timestamp."""
        return f"watchly:last_profile_build:{token}:{content_type}"

    # Library Items Methods

    async def get_library_items(self, token: str) -> dict[str, Any] | None:
        """
        Get cached library items for a user.

        Args:
            token: User token

        Returns:
            Library items dictionary, or None if not cached
        """
        key = self._library_items_key(token)
        cached = await redis_service.get(key)

        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode cached library items for {redact_token(token)}...: {e}")
                return None

        return None

    async def set_library_items(self, token: str, library_items: dict[str, Any]) -> None:
        """
        Cache library items for a user.

        Args:
            token: User token
            library_items: Library items dictionary to cache
        """
        # Refuse to cache an empty library. Upstream clients (Trakt/Stremio) swallow
        # auth and rate-limit failures and return an empty-but-well-formed dict rather
        # than raising. Caching that -- with no TTL -- pins the account into permanent
        # trending/TMDB fallback until the key is manually deleted.
        _li = library_items or {}
        if not any(_li.get(k) for k in ("watched", "loved", "liked", "added")):
            logger.warning(
                f"[{redact_token(token)}...] Library has no items; refusing to cache "
                "(likely upstream auth or rate-limit failure)"
            )
            return

        key = self._library_items_key(token)
        await redis_service.set(key, json.dumps(library_items), DERIVED_CACHE_TTL_SECONDS)
        logger.debug(f"[{redact_token(token)}...] Cached library items")

        # Preserve this profile's last-known-good published catalogs.
        # Library changes make only this token's catalogs dirty; replacement
        # catalogs will be generated separately in the background.
        await self.mark_catalogs_dirty(token)

    async def invalidate_library_items(self, token: str) -> None:
        """
        Invalidate cached library items for a user.

        Args:
            token: User token
        """
        key = self._library_items_key(token)
        await redis_service.delete(key)
        logger.debug(f"[{redact_token(token)}...] Invalidated library items cache")

    # Profile Methods

    async def get_profile(self, token: str, content_type: str) -> TasteProfile | None:
        """
        Get cached profile for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)

        Returns:
            TasteProfile instance, or None if not cached
        """
        key = self._profile_key(token, content_type)
        cached = await redis_service.get(key)

        if cached:
            try:
                return TasteProfile.model_validate_json(cached)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to decode cached profile for {redact_token(token)}.../{content_type}: {e}")
                return None

        return None

    async def set_profile(self, token: str, content_type: str, profile: TasteProfile) -> None:
        """
        Cache profile for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)
            profile: TasteProfile instance to cache
        """
        key = self._profile_key(token, content_type)
        await redis_service.set(key, profile.model_dump_json(), DERIVED_CACHE_TTL_SECONDS)
        logger.debug(f"[{redact_token(token)}...] Cached profile for {content_type}")

    async def invalidate_profile(self, token: str, content_type: str) -> None:
        """
        Invalidate cached profile for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)
        """
        key = self._profile_key(token, content_type)
        await redis_service.delete(key)
        logger.debug(f"[{redact_token(token)}...] Invalidated profile cache for {content_type}")

    # Watched Sets Methods

    async def get_watched_sets(self, token: str, content_type: str) -> tuple[set[int], set[str]] | None:
        """
        Get cached watched sets for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)

        Returns:
            Tuple of (watched_tmdb set, watched_imdb set), or None if not cached
        """
        key = self._watched_sets_key(token, content_type)
        cached = await redis_service.get(key)

        if cached:
            try:
                data = json.loads(cached)
                watched_tmdb = set(data.get("watched_tmdb", []))
                watched_imdb = set(data.get("watched_imdb", []))
                return (watched_tmdb, watched_imdb)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to decode cached watched sets for {redact_token(token)}.../{content_type}: {e}")
                return None

        return None

    async def set_watched_sets(
        self,
        token: str,
        content_type: str,
        watched_tmdb: set[int],
        watched_imdb: set[str],
    ) -> None:
        """
        Cache watched sets for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)
            watched_tmdb: Set of watched TMDB IDs
            watched_imdb: Set of watched IMDb IDs
        """
        key = self._watched_sets_key(token, content_type)
        data = {
            "watched_tmdb": list(watched_tmdb),
            "watched_imdb": list(watched_imdb),
        }
        await redis_service.set(key, json.dumps(data), DERIVED_CACHE_TTL_SECONDS)
        logger.debug(f"[{redact_token(token)}...] Cached watched sets for {content_type}")

    async def invalidate_watched_sets(self, token: str, content_type: str) -> None:
        """
        Invalidate cached watched sets for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)
        """
        key = self._watched_sets_key(token, content_type)
        await redis_service.delete(key)
        logger.debug(f"[{redact_token(token)}...] Invalidated watched sets cache for {content_type}")

    # Combined Methods

    async def get_profile_and_watched_sets(
        self, token: str, content_type: str
    ) -> tuple[TasteProfile | None, set[int], set[str]] | None:
        """
        Get both cached profile and watched sets for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)

        Returns:
            Tuple of (profile, watched_tmdb, watched_imdb), or None if either is not cached.
            Returns None if either profile or watched sets are missing.
        """
        profile = await self.get_profile(token, content_type)
        watched_sets = await self.get_watched_sets(token, content_type)

        if profile is None or watched_sets is None:
            return None

        watched_tmdb, watched_imdb = watched_sets
        return (profile, watched_tmdb, watched_imdb)

    # Library Change Detection Methods

    async def has_library_changed(self, token: str, content_type: str, library_items: list) -> bool:
        """
        Check if library has changed since last profile build.

        Args:
            token: User token
            content_type: Content type (movie or series)
            library_items: Current library items list

        Returns:
            True if library has changed, False otherwise
        """
        # Create hash of current library item IDs
        current_ids = [item.get("_id", item.get("id", "")) for item in library_items]
        current_hash = hashlib.md5("".join(sorted(current_ids)).encode()).hexdigest()

        # Compare with stored hash
        stored_hash = await redis_service.get(self._library_hash_key(token, content_type))

        if stored_hash is None:
            # No stored hash, consider it changed
            return True

        return current_hash != stored_hash.decode() if isinstance(stored_hash, bytes) else current_hash != stored_hash

    async def update_library_hash(self, token: str, content_type: str, library_items: list) -> None:
        """
        Update the stored library hash after successful profile build.

        Args:
            token: User token
            content_type: Content type (movie or series)
            library_items: Current library items list
        """
        current_ids = [item.get("_id", item.get("id", "")) for item in library_items]
        current_hash = hashlib.md5("".join(sorted(current_ids)).encode()).hexdigest()

        hash_key = self._library_hash_key(token, content_type)
        build_time_key = self._last_profile_build_key(token, content_type)

        # Store hash and build timestamp
        await redis_service.set(hash_key, current_hash, DERIVED_CACHE_TTL_SECONDS)
        await redis_service.set(build_time_key, str(time.time()), DERIVED_CACHE_TTL_SECONDS)

        logger.debug(f"[{redact_token(token)}...] Updated library hash for {content_type}")

    async def get_last_profile_build_time(self, token: str, content_type: str) -> int | None:
        """
        Get the timestamp of the last profile build.

        Args:
            token: User token
            content_type: Content type (movie or series)

        Returns:
            Unix timestamp of last build, or None if never built
        """
        build_time = await redis_service.get(self._last_profile_build_key(token, content_type))
        if build_time is None:
            return None

        try:
            return int(float(build_time.decode() if isinstance(build_time, bytes) else build_time))
        except (ValueError, TypeError):
            return None

    async def set_profile_and_watched_sets(
        self,
        token: str,
        content_type: str,
        profile: TasteProfile | None,
        watched_tmdb: set[int],
        watched_imdb: set[str],
    ) -> None:
        """
        Cache both profile and watched sets for a user and content type.

        Args:
            token: User token
            content_type: Content type (movie or series)
            profile: TasteProfile instance to cache (can be None)
            watched_tmdb: Set of watched TMDB IDs
            watched_imdb: Set of watched IMDb IDs
        """
        if profile:
            await self.set_profile(token, content_type, profile)
        await self.set_watched_sets(token, content_type, watched_tmdb, watched_imdb)

        # Preserve this profile's last-known-good published catalogs.
        # Profile changes dirty only this token's catalogs and must never
        # delete another profile's published snapshot.
        await self.mark_catalogs_dirty(token)

    # Invalidation Methods

    async def invalidate_all_user_data(self, token: str) -> None:
        """
        Invalidate all cached data for a user (library items, profiles, watched sets, catalogs).

        Args:
            token: User token
        """
        await self.invalidate_library_items(token)
        for content_type in ["movie", "series"]:
            await self.invalidate_profile(token, content_type)
            await self.invalidate_watched_sets(token, content_type)
        await self.invalidate_all_catalogs(token)
        logger.debug(f"[{redact_token(token)}...] Invalidated all user data cache")

    @staticmethod
    def _catalog_dirty_key(token: str) -> str:
        # Deliberately token-scoped for complete profile isolation.
        return f"watchly:catalog_dirty:{token}"

    async def mark_catalogs_dirty(self, token: str) -> None:
        """Mark this profile's published catalogs as needing regeneration.

        Existing catalog payloads remain intact so clients can continue using
        the last-known-good published snapshot while replacements are built.
        """
        # Use a nanosecond revision rather than whole seconds so several
        # profile/library changes in rapid succession cannot collapse into one
        # generation marker.
        dirty_revision = time.time_ns()
        await redis_service.set(
            self._catalog_dirty_key(token),
            str(dirty_revision),
        )
        logger.debug(
            f"[{redact_token(token)}...] Marked published catalogs dirty"
        )


    async def get_catalog_dirty_revision(self, token: str) -> int | None:
        """Return this profile's most recent catalog-dirty revision."""
        value = await redis_service.get(self._catalog_dirty_key(token))
        if value is None:
            return None

        try:
            if isinstance(value, bytes):
                value = value.decode()
            return int(value)
        except (TypeError, ValueError):
            return None

    async def get_catalog(
        self,
        token: str,
        type: str,
        id: str,
    ) -> tuple[dict[str, Any], int, int | None] | None:
        """
        Get cached catalog for a user and content type.

        Args:
            token: User token
            type: Content type (movie or series)
            id: Catalog ID

        Returns:
            Tuple of (catalog_data, timestamp) or None if not found
        """
        key = CATALOG_KEY.format(token=token, type=type, id=id)
        cached = await redis_service.get(key)
        if cached:
            try:
                data = json.loads(cached)
                # Handle new format with timestamp wrapper
                if "data" in data and "created_at" in data:
                    return (
                        data["data"],
                        data["created_at"],
                        data.get("source_revision"),
                    )
                # Handle legacy format (raw catalog dict). No source revision
                # means a later dirty marker will correctly force regeneration.
                return data, 0, None
            except json.JSONDecodeError:
                return None
        return None

    async def get_raw_catalog(
        self,
        token: str,
        type: str,
        id: str,
    ) -> tuple[dict[str, Any], int, int | None] | None:
        """Get the undeduped source snapshot for a catalog, if available."""
        key = RAW_CATALOG_KEY.format(token=token, type=type, id=id)
        cached = await redis_service.get(key)
        if cached:
            try:
                data = json.loads(cached)
                if "data" in data and "created_at" in data:
                    return (
                        data["data"],
                        data["created_at"],
                        data.get("source_revision"),
                    )
                return data, 0, None
            except json.JSONDecodeError:
                return None
        return None

    async def set_raw_catalog(
        self,
        token: str,
        type: str,
        id: str,
        catalog: dict[str, Any],
        ttl: int | None = None,
        source_revision: int | None = None,
    ) -> None:
        """Store the undeduped source snapshot used for coordinated row balancing."""
        key = RAW_CATALOG_KEY.format(token=token, type=type, id=id)
        wrapped_data = {
            "data": catalog,
            "created_at": int(time.time()),
            "source_revision": source_revision,
        }
        await redis_service.set(key, json.dumps(wrapped_data), ttl)
        logger.debug(f"[{redact_token(token)}...] Cached raw catalog for {type}/{id}")

    async def set_catalog(
        self,
        token: str,
        type: str,
        id: str,
        catalog: dict[str, Any],
        ttl: int | None = None,
        source_revision: int | None = None,
    ) -> None:
        """
        Cache catalog for a user and content type.

        Args:
            token: User token
            type: Content type (movie or series)
            id: Catalog ID
            catalog: Catalog dictionary to cache
            ttl: Time to live for the cache (in seconds)
        """
        key = CATALOG_KEY.format(token=token, type=type, id=id)
        # Store with timestamp for stale-while-revalidate logic
        wrapped_data = {
            "data": catalog,
            "created_at": int(time.time()),
            # Revision of the profile/library snapshot used to build this
            # published payload. A later token-scoped dirty revision means the
            # payload stays visible but needs a background replacement.
            "source_revision": source_revision,
        }
        await redis_service.set(key, json.dumps(wrapped_data), ttl)
        logger.debug(f"[{redact_token(token)}...] Cached catalog for {type}/{id}")

    async def get_staged_catalog(
        self,
        token: str,
        type: str,
        id: str,
    ) -> tuple[dict[str, Any], int, int | None] | None:
        """Get a candidate-generation catalog without exposing it to clients."""
        key = STAGED_CATALOG_KEY.format(token=token, type=type, id=id)
        cached = await redis_service.get(key)
        if not cached:
            return None
        try:
            data = json.loads(cached)
            if "data" in data and "created_at" in data:
                return (
                    data["data"],
                    data["created_at"],
                    data.get("source_revision"),
                )
            return data, 0, None
        except json.JSONDecodeError:
            return None

    async def set_staged_catalog(
        self,
        token: str,
        type: str,
        id: str,
        catalog: dict[str, Any],
        source_revision: int | None = None,
    ) -> None:
        """Store a candidate-generation catalog outside the published namespace."""
        key = STAGED_CATALOG_KEY.format(token=token, type=type, id=id)
        wrapped_data = {
            "data": catalog,
            "created_at": int(time.time()),
            "source_revision": source_revision,
        }
        await redis_service.set(
            key,
            json.dumps(wrapped_data),
            30 * 24 * 60 * 60,
        )

    async def get_staged_raw_catalog(
        self,
        token: str,
        type: str,
        id: str,
    ) -> tuple[dict[str, Any], int, int | None] | None:
        """Get the undeduped candidate-generation source snapshot."""
        key = STAGED_RAW_CATALOG_KEY.format(token=token, type=type, id=id)
        cached = await redis_service.get(key)
        if not cached:
            return None
        try:
            data = json.loads(cached)
            if "data" in data and "created_at" in data:
                return (
                    data["data"],
                    data["created_at"],
                    data.get("source_revision"),
                )
            return data, 0, None
        except json.JSONDecodeError:
            return None

    async def set_staged_raw_catalog(
        self,
        token: str,
        type: str,
        id: str,
        catalog: dict[str, Any],
        source_revision: int | None = None,
    ) -> None:
        """Store an undeduped candidate source snapshot outside the live namespace."""
        key = STAGED_RAW_CATALOG_KEY.format(token=token, type=type, id=id)
        wrapped_data = {
            "data": catalog,
            "created_at": int(time.time()),
            "source_revision": source_revision,
        }
        await redis_service.set(
            key,
            json.dumps(wrapped_data),
            30 * 24 * 60 * 60,
        )

    async def clear_staged_catalogs(self, token: str) -> None:
        """Remove all unpublished candidate catalog payloads for one token."""
        await redis_service.delete_by_pattern(
            f"watchly:catalog_staged:{token}:*"
        )
        await redis_service.delete_by_pattern(
            f"watchly:catalog_staged_raw:{token}:*"
        )

    async def promote_prepared_generation(
        self,
        token: str,
        manifest: dict[str, Any],
        _watch_retry: int = 0,
    ) -> bool:
        """Atomically publish one complete staged generation if it is still current."""
        prepared_key = f"watchly:manifest:prepared:{token}"
        dirty_key = self._catalog_dirty_key(token)
        staged = []
        watch_keys = [prepared_key, dirty_key]

        for catalog in manifest.get("catalogs", []):
            content_type = catalog.get("type")
            catalog_id = catalog.get("id")
            if not content_type or not catalog_id:
                return False
            staged_key = STAGED_CATALOG_KEY.format(token=token, type=content_type, id=catalog_id)
            staged_raw_key = STAGED_RAW_CATALOG_KEY.format(token=token, type=content_type, id=catalog_id)
            staged.append((
                CATALOG_KEY.format(token=token, type=content_type, id=catalog_id),
                RAW_CATALOG_KEY.format(token=token, type=content_type, id=catalog_id),
                staged_key,
                staged_raw_key,
            ))
            watch_keys.extend((staged_key, staged_raw_key))

        client = await redis_service.get_client()
        try:
            async with client.pipeline(transaction=True) as pipe:
                await pipe.watch(*watch_keys)

                prepared_value = await pipe.get(prepared_key)
                if prepared_value is None:
                    return False
                try:
                    current_prepared = json.loads(prepared_value)
                except (TypeError, json.JSONDecodeError):
                    current_prepared = None
                if current_prepared != manifest:
                    return False

                dirty_value = await pipe.get(dirty_key)
                dirty_valid = True
                try:
                    current_revision = int(dirty_value) if dirty_value is not None else None
                except (TypeError, ValueError):
                    dirty_valid = False
                    current_revision = None

                payloads = []
                revisions = set()
                generation_valid = dirty_valid
                for live_key, live_raw_key, staged_key, staged_raw_key in staged:
                    staged_value = await pipe.get(staged_key)
                    staged_raw_value = await pipe.get(staged_raw_key)
                    if staged_value is None or staged_raw_value is None:
                        generation_valid = False
                        continue
                    try:
                        staged_wrapped = json.loads(staged_value)
                        staged_raw_wrapped = json.loads(staged_raw_value)
                        staged_revision = staged_wrapped.get("source_revision")
                        staged_raw_revision = staged_raw_wrapped.get("source_revision")
                    except (AttributeError, TypeError, json.JSONDecodeError):
                        generation_valid = False
                        continue
                    if staged_revision != staged_raw_revision:
                        generation_valid = False
                    revisions.add(staged_revision)
                    payloads.append((
                        live_key, live_raw_key, staged_key, staged_raw_key,
                        staged_value, staged_raw_value,
                    ))

                if len(revisions) != 1 or next(iter(revisions), None) != current_revision:
                    generation_valid = False

                pipe.multi()
                if not generation_valid or len(payloads) != len(staged):
                    pipe.delete(prepared_key)
                    for _, _, staged_key, staged_raw_key in staged:
                        pipe.delete(staged_key, staged_raw_key)
                    await pipe.execute()
                    logger.info(
                        f"[{redact_token(token)}...] Discarded stale/incomplete prepared "
                        "generation without changing published catalogs"
                    )
                    return False

                for (
                    live_key, live_raw_key, staged_key, staged_raw_key,
                    staged_value, staged_raw_value,
                ) in payloads:
                    pipe.set(live_key, staged_value)
                    pipe.set(live_raw_key, staged_raw_value)
                    pipe.delete(staged_key, staged_raw_key)

                wrapped_manifest = json.dumps({
                    "manifest": manifest,
                    "created_at": int(time.time()),
                })
                pipe.setex(
                    f"watchly:manifest:{token}",
                    settings.MANIFEST_CACHE_TTL_SECONDS,
                    wrapped_manifest,
                )
                pipe.set(
                    f"watchly:manifest:active:{token}",
                    json.dumps(manifest),
                    ex=30 * 24 * 60 * 60,
                )
                pipe.delete(prepared_key)
                await pipe.execute()
                return True
        except WatchError:
            if _watch_retry < 2:
                logger.info(
                    f"[{redact_token(token)}...] Prepared generation changed during "
                    "atomic promotion; retrying guarded validation"
                )
                return await self.promote_prepared_generation(
                    token,
                    manifest,
                    _watch_retry + 1,
                )

            logger.warning(
                f"[{redact_token(token)}...] Prepared generation kept changing during "
                "atomic promotion; published generation left untouched"
            )
            return False

    async def get_manifest(self, token: str) -> dict | None:
        """Return cached manifest if still within the manifest cache TTL."""
        key = f"watchly:manifest:{token}"
        try:
            data = await redis_service.get(key)
            if data:
                wrapped = json.loads(data)
                age = int(time.time()) - wrapped.get("created_at", 0)
                if age < settings.MANIFEST_CACHE_TTL_SECONDS:
                    return wrapped["manifest"]
        except Exception as e:
            logger.warning(f"[{redact_token(token)}...] Failed to get cached manifest: {e}")
        return None

    async def set_manifest(self, token: str, manifest: dict) -> None:
        """Cache the manifest for MANIFEST_CACHE_TTL_SECONDS."""
        key = f"watchly:manifest:{token}"
        try:
            wrapped = {"manifest": manifest, "created_at": int(time.time())}
            await redis_service.set(key, json.dumps(wrapped), settings.MANIFEST_CACHE_TTL_SECONDS)
            await self.set_active_manifest(token, manifest)
            logger.debug(f"[{redact_token(token)}...] Cached manifest (TTL: {settings.MANIFEST_CACHE_TTL_SECONDS}s)")
        except Exception as e:
            logger.warning(f"[{redact_token(token)}...] Failed to cache manifest: {e}")

    async def get_active_manifest(self, token: str) -> dict | None:
        key = f"watchly:manifest:active:{token}"
        try:
            data = await redis_service.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"[{redact_token(token)}...] Failed to get active manifest snapshot: {e}")
            return None

    async def set_active_manifest(self, token: str, manifest: dict) -> None:
        key = f"watchly:manifest:active:{token}"
        try:
            await redis_service.set(key, json.dumps(manifest), 30 * 24 * 60 * 60)
        except Exception as e:
            logger.warning(f"[{redact_token(token)}...] Failed to store active manifest snapshot: {e}")

    async def get_prepared_manifest(self, token: str) -> dict | None:
        key = f"watchly:manifest:prepared:{token}"
        try:
            data = await redis_service.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"[{redact_token(token)}...] Failed to get prepared manifest: {e}")
            return None

    async def set_prepared_manifest(self, token: str, manifest: dict) -> None:
        key = f"watchly:manifest:prepared:{token}"
        await redis_service.set(key, json.dumps(manifest), 30 * 24 * 60 * 60)
        logger.info(
            f"[{redact_token(token)}...] Staged fully prewarmed next manifest "
            f"with {len(manifest.get('catalogs', []))} catalogs"
        )

    async def clear_prepared_manifest(self, token: str) -> None:
        key = f"watchly:manifest:prepared:{token}"
        await redis_service.delete(key)

    async def invalidate_manifest(self, token: str) -> None:
        """Invalidate cached manifest so it regenerates on next request."""
        key = f"watchly:manifest:{token}"
        try:
            await redis_service.delete(key)
            await self.clear_prepared_manifest(token)
            await self.clear_staged_catalogs(token)
            logger.debug(f"[{redact_token(token)}...] Invalidated manifest cache")
        except Exception as e:
            logger.warning(f"[{redact_token(token)}...] Failed to invalidate manifest: {e}")

    async def invalidate_catalog(self, token: str, type: str, id: str) -> None:
        """
        Invalidate cached catalog for a user and content type.

        Args:
            token: User token
            type: Content type (movie or series)
            id: Catalog ID
        """
        key = CATALOG_KEY.format(token=token, type=type, id=id)
        raw_key = RAW_CATALOG_KEY.format(token=token, type=type, id=id)
        staged_key = STAGED_CATALOG_KEY.format(token=token, type=type, id=id)
        staged_raw_key = STAGED_RAW_CATALOG_KEY.format(token=token, type=type, id=id)
        await redis_service.delete(key)
        await redis_service.delete(raw_key)
        await redis_service.delete(staged_key)
        await redis_service.delete(staged_raw_key)
        logger.debug(f"[{redact_token(token)}...] Invalidated catalog cache for {type}/{id}")

    async def invalidate_all_catalogs(self, token: str) -> None:
        """
        Invalidate all cached catalogs for a user.

        This should be called when user data (library items, profiles) is updated
        to ensure catalogs are regenerated with fresh data.

        Args:
            token: User token
        """
        pattern = f"watchly:catalog:{token}:*"
        raw_pattern = f"watchly:catalog_raw:{token}:*"
        staged_pattern = f"watchly:catalog_staged:{token}:*"
        staged_raw_pattern = f"watchly:catalog_staged_raw:{token}:*"
        deleted_count = await redis_service.delete_by_pattern(pattern)
        deleted_count += await redis_service.delete_by_pattern(raw_pattern)
        deleted_count += await redis_service.delete_by_pattern(staged_pattern)
        deleted_count += await redis_service.delete_by_pattern(staged_raw_pattern)
        # This remains the explicit destructive-reset path, so clear this
        # profile's dirty marker as part of the same token-scoped reset.
        await redis_service.delete(self._catalog_dirty_key(token))
        if deleted_count > 0:
            logger.debug(f"[{redact_token(token)}...] Invalidated {deleted_count} catalog cache(s)")
        else:
            logger.debug(f"[{redact_token(token)}...] No catalog caches found to invalidate")


user_cache = UserCacheService()
