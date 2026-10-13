from fastapi import APIRouter, HTTPException, Response
from loguru import logger

from app.core.security import redact_token
from app.services.recommendation.catalog_service import catalog_service

router = APIRouter()


@router.get("/{token}/catalog/{type}/{id}.json")
@router.get("/{token}/catalog/{type}/{id}/{extra}.json")
async def get_catalog(response: Response, type: str, id: str, token: str, extra: str | None = None) -> dict:
    if type not in ("movie", "series"):
        raise HTTPException(status_code=400, detail="Invalid content type. Must be 'movie' or 'series'.")

    if len(token) > 80:  # Stremio tokens are ~24 chars; Trakt hashed tokens are 40 chars. 80 is a safe upper bound.
        raise HTTPException(status_code=400, detail="Invalid token.")

    try:
        # Delegate to catalog service facade
        recommendations, headers = await catalog_service.get_catalog(token, type, id)

        # Stremio pagination is expressed as an extra path segment such as
        # skip=19. Keep pagination outside CatalogService so cached/shared
        # catalogs always contain the canonical full result.
        skip = 0
        if extra:
            for part in extra.split("&"):
                name, separator, value = part.partition("=")
                if name == "skip" and separator:
                    try:
                        parsed_skip = int(value)
                    except ValueError:
                        break
                    if parsed_skip >= 0:
                        skip = parsed_skip
                    break

        if skip and recommendations:
            recommendations = dict(recommendations)
            if "metas" in recommendations:
                recommendations["metas"] = list(recommendations.get("metas") or [])[skip:]
            elif "meta" in recommendations:
                recommendations["meta"] = list(recommendations.get("meta") or [])[skip:]

        # Set response headers
        for key, value in headers.items():
            response.headers[key] = value

        # if recommendations are none or empty, then set cache header to no-cache
        if recommendations and not recommendations.get("meta"):
            response.headers["Cache-Control"] = "no-cache"

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{redact_token(token)}] Error fetching catalog for {type}/{id}: {e}")
        raise HTTPException(status_code=500, detail=f"Something went wrong. Please try again. Error: {e}")
