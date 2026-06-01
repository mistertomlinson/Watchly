from fastapi import APIRouter, HTTPException
import httpx
from loguru import logger

from app.api.models.validation import BaseValidationInput, BaseValidationResponse, PosterRatingValidationInput
from app.services.poster_ratings.factory import PosterProvider, poster_ratings_factory
from app.services.simkl import simkl_service
from app.services.tmdb.client import TMDBClient

router = APIRouter(tags=["Validation"])


@router.post("/gemini/validation")
async def validate_openrouter_api_key(data: BaseValidationInput) -> BaseValidationResponse:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {data.api_key.strip()}"},
            )
            if response.status_code == 200:
                return BaseValidationResponse(valid=True, message="OpenRouter API key is valid")
            else:
                return BaseValidationResponse(valid=False, message="Invalid OpenRouter API key")
    except Exception as e:
        logger.debug(f"OpenRouter API key validation failed: {e}")
        return BaseValidationResponse(valid=False, message="Invalid OpenRouter API key")


@router.post("/tmdb/validation")
async def validate_tmdb_api_key(data: BaseValidationInput) -> BaseValidationResponse:
    try:
        client = TMDBClient(api_key=data.api_key.strip(), language="en-US")
        await client.get("/configuration")
        await client.close()
        return BaseValidationResponse(valid=True, message="TMDB API key is valid")
    except Exception as e:
        logger.debug(f"TMDB API key validation failed: {e}")
        return BaseValidationResponse(valid=False, message="Invalid TMDB API key")


@router.post("/poster-rating/validate")
async def validate_poster_rating_api_key(payload: PosterRatingValidationInput) -> BaseValidationResponse:
    if not payload.api_key or not payload.api_key.strip():
        return BaseValidationResponse(valid=False, message="API key cannot be empty")

    try:
        provider_enum = PosterProvider(payload.provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {payload.provider}")

    try:
        if provider_enum == PosterProvider.RPDB:
            is_valid = await poster_ratings_factory.rpdb_service.validate_api_key(payload.api_key.strip())
        elif provider_enum == PosterProvider.TOP_POSTERS:
            is_valid = await poster_ratings_factory.top_posters_service.validate_api_key(payload.api_key.strip())
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {payload.provider}")

        if is_valid:
            return BaseValidationResponse(valid=True, message="API key is valid")
        return BaseValidationResponse(valid=False, message="Invalid API key")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation failed due to an internal error.")


@router.post("/simkl/validation")
async def validate_simkl_api_key(data: BaseValidationInput) -> BaseValidationResponse:
    try:
        response = await simkl_service.get_trending(data.api_key)
        if response:
            return BaseValidationResponse(valid=True, message="Valid API Key")
        return BaseValidationResponse(valid=False, message="Invalid API Key")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation failed due to an internal error.")
