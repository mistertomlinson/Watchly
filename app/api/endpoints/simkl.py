import html
import json
import secrets
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security import redact_token
from app.core.settings import CatalogConfig, PosterRatingConfig, UserSettings, get_default_settings
from app.services.manifest import manifest_service
from app.services.simkl_provider import SimklApiClient, SimklLibraryProvider
from app.services.token_store import token_store

router = APIRouter(prefix="/tokens/simkl", tags=["simkl"])
_OAUTH_STATE_TTL = 600


class SimklTokenRequest(BaseModel):
    simkl_access_token: str
    catalogs: list[CatalogConfig] | None = None
    language: str = "en-US"
    poster_rating: PosterRatingConfig | None = None
    excluded_movie_genres: list[str] = Field(default_factory=list)
    excluded_series_genres: list[str] = Field(default_factory=list)
    popularity: str = "balanced"
    year_min: int = 2010
    year_max: int = 2026
    sorting_order: str = "default"
    openrouter_api_key: str | None = None
    tmdb_api_key: str | None = None


class SimklTokenResponse(BaseModel):
    token: str
    manifestUrl: str
    expiresInSeconds: int | None = None


def _require_config() -> tuple[str, str]:
    if not settings.SIMKL_CLIENT_ID or not settings.SIMKL_CLIENT_SECRET:
        raise HTTPException(status_code=503, detail="Simkl integration is not configured on this server")
    return settings.SIMKL_CLIENT_ID, settings.SIMKL_CLIENT_SECRET


async def _store_state(state: str) -> None:
    from app.services.redis_service import redis_service
    await redis_service.set(f"watchly:oauth_state:simkl:{state}", state, _OAUTH_STATE_TTL)


async def _consume_state(state: str) -> bool:
    from app.services.redis_service import redis_service
    key = f"watchly:oauth_state:simkl:{state}"
    value = await redis_service.get(key)
    if value:
        await redis_service.delete(key)
        return True
    return False


@router.get("/config")
async def simkl_config():
    configured = bool(settings.SIMKL_CLIENT_ID and settings.SIMKL_CLIENT_SECRET)
    return {"configured": configured, "client_id": settings.SIMKL_CLIENT_ID if configured else None}


@router.get("/authorize")
async def simkl_authorize():
    client_id, _ = _require_config()
    state = secrets.token_urlsafe(16)
    await _store_state(state)
    redirect_uri = f"{settings.HOST_NAME}/tokens/simkl/callback"
    query = urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    })
    return {"url": f"https://simkl.com/oauth/authorize?{query}", "state": state}


@router.get("/callback", response_class=HTMLResponse)
async def simkl_callback(code: str | None = None, state: str | None = None, error: str | None = None):
    if error or not code:
        return HTMLResponse(_popup_html(False, error=error or "Authorization cancelled"))
    if not state or not await _consume_state(state):
        return HTMLResponse(_popup_html(False, error="Invalid OAuth state. Please try again."))

    client_id, client_secret = _require_config()
    redirect_uri = f"{settings.HOST_NAME}/tokens/simkl/callback"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                "https://api.simkl.com/oauth/token",
                json={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": redirect_uri,
                    "code": code,
                },
            )
            response.raise_for_status()
            token_data = response.json()
    except Exception as exc:
        logger.error(f"Simkl callback token exchange failed: {exc}")
        return HTMLResponse(_popup_html(False, error="Token exchange failed"))

    access_token = token_data.get("access_token")
    if not access_token:
        return HTMLResponse(_popup_html(False, error="No access token received"))
    return HTMLResponse(_popup_html(True, access_token=access_token))


async def _identity(access_token: str) -> tuple[str, str, dict[str, Any]]:
    client_id, _ = _require_config()
    client = SimklApiClient(client_id, access_token)
    try:
        user = await client.get_user()
    finally:
        await client.close()
    account = user.get("account") if isinstance(user.get("account"), dict) else {}
    profile = user.get("user") if isinstance(user.get("user"), dict) else {}
    public_profile = user.get("profile") if isinstance(user.get("profile"), dict) else {}

    # Preserve Simkl's stable account ID internally while locating the public
    # display name in any of the profile containers returned by /users/settings.
    stable_id = str(
        account.get("id")
        or profile.get("id")
        or public_profile.get("id")
        or user.get("id")
        or account.get("username")
        or profile.get("username")
        or public_profile.get("username")
        or user.get("username")
        or "simkl_user"
    )

    containers = [profile, public_profile, account, user]
    for container in list(containers):
        for nested_key in ("user", "profile", "account"):
            nested = container.get(nested_key)
            if isinstance(nested, dict) and nested not in containers:
                containers.append(nested)

    display_name = "Simkl User"
    for field in ("name", "username", "nickname", "display_name", "login"):
        for container in containers:
            value = container.get(field)
            candidate = str(value).strip() if value is not None else ""
            if candidate and not candidate.isdigit():
                display_name = candidate
                break
        if display_name != "Simkl User":
            break

    if display_name == "Simkl User":
        logger.info(
            "Simkl profile did not expose a readable name; "
            f"top-level keys={sorted(user.keys())}, "
            f"account keys={sorted(account.keys())}, "
            f"user-profile keys={sorted(profile.keys())}"
        )

    user_id = f"simkl:{stable_id}"
    return user_id, display_name, user


@router.post("/identity")
async def simkl_identity(payload: SimklTokenRequest):
    try:
        user_id, username, user = await _identity(payload.simkl_access_token)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Simkl access token: {exc}") from exc
    token = token_store.get_token_from_user_id(user_id)
    user_data = await token_store.get_user_data(token)
    response: dict[str, Any] = {
        "user_id": user_id,
        "username": username,
        "display": username,
        "exists": bool(user_data),
    }
    if user_data:
        response["settings"] = user_data.get("settings", {})
    return response


@router.post("/", response_model=SimklTokenResponse)
async def create_simkl_token(payload: SimklTokenRequest, request: Request):
    try:
        user_id, username, user = await _identity(payload.simkl_access_token)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Simkl access token: {exc}") from exc

    token = token_store.get_token_from_user_id(user_id)
    existing_data = await token_store.get_user_data(token)
    defaults = get_default_settings()
    user_settings = UserSettings(
        language=payload.language or defaults.language,
        catalogs=payload.catalogs or defaults.catalogs,
        poster_rating=payload.poster_rating,
        excluded_movie_genres=payload.excluded_movie_genres,
        excluded_series_genres=payload.excluded_series_genres,
        year_min=payload.year_min,
        year_max=payload.year_max,
        popularity=payload.popularity,
        sorting_order=payload.sorting_order,
        simkl_api_key=settings.SIMKL_CLIENT_ID,
        openrouter_api_key=payload.openrouter_api_key,
        tmdb_api_key=payload.tmdb_api_key,
    )
    stored: dict[str, Any] = {
        "auth_provider": "simkl",
        "authKey": payload.simkl_access_token,
        "simkl_username": username,
        "email": user.get("email") or username,
        "settings": user_settings.model_dump(),
        "last_updated": existing_data.get("last_updated") if existing_data else datetime.now(timezone.utc).isoformat(),
    }
    token = await token_store.store_user_data(user_id, stored)
    logger.info(f"[{redact_token(token)}] Simkl account {'updated' if existing_data else 'created'} for {user_id}")

    client = SimklApiClient(settings.SIMKL_CLIENT_ID, payload.simkl_access_token)
    try:
        library = await SimklLibraryProvider(client).get_library_items()
        await manifest_service.cache_library_and_profiles_from_items(library, user_settings, token)
    except Exception as exc:
        logger.warning(f"[{redact_token(token)}] Failed to pre-cache Simkl library: {exc}")
    finally:
        await client.close()

    manifest_url = f"{settings.HOST_NAME}/{token}/manifest.json"
    expires = settings.TOKEN_TTL_SECONDS if settings.TOKEN_TTL_SECONDS > 0 else None
    return SimklTokenResponse(token=token, manifestUrl=manifest_url, expiresInSeconds=expires)


def _popup_html(success: bool, error: str | None = None, access_token: str | None = None) -> str:
    if success:
        payload = {"type": "simkl_auth_success", "access_token": access_token}
    else:
        payload = {"type": "simkl_auth_error", "error": error or "Unknown error"}
    payload_js = json.dumps(payload).replace("</", "<\\/")
    body = "Authorization successful! You can close this window." if success else f"Authorization failed: {html.escape(error or 'Unknown error')}"
    return f"""<!DOCTYPE html><html><head><title>Simkl Authorization</title></head><body>
<p style='font-family:sans-serif;text-align:center;margin-top:3rem'>{body}</p>
<script>try{{if(window.opener)window.opener.postMessage({payload_js},window.location.origin)}}catch(e){{}}setTimeout(function(){{window.close()}},1000);</script>
</body></html>"""
