from fastapi import Response
from fastapi.responses import RedirectResponse
from fastapi.routing import APIRouter

from app.services.manifest import manifest_service

router = APIRouter()


@router.get("/manifest.json")
async def manifest():
    """Get base manifest for unauthenticated users."""
    manifest = manifest_service.get_base_manifest()
    # since user is not logged in, return empty catalogs
    manifest["catalogs"] = []
    return manifest


@router.get("/{token}/manifest.json")
async def manifest_token(token: str):
    """Get manifest for authenticated user."""
    return await manifest_service.get_manifest_for_token(token)


@router.get("/{token}")
async def manifest_token_redirect(token: str):
    """Redirect bare token URL to manifest.json.
    Some clients (e.g. Nuvio) strip /manifest.json when storing the addon URL
    and later re-fetch without it. This redirect ensures they still get the manifest.
    """
    return RedirectResponse(url=f"/{token}/manifest.json", status_code=301)
