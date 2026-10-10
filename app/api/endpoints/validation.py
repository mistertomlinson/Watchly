import re

from fastapi import APIRouter, HTTPException
import httpx
from loguru import logger

from app.api.models.validation import BaseValidationInput, BaseValidationResponse, PosterRatingValidationInput
from app.services.poster_ratings.factory import PosterProvider, poster_ratings_factory
from app.services.tmdb.client import TMDBClient

router = APIRouter(tags=["Validation"])


@router.post("/gemini/validation")
async def validate_ai_api_key(data: BaseValidationInput) -> BaseValidationResponse:
    api_key = data.api_key.strip()
    if not api_key:
        return BaseValidationResponse(valid=False, message="AI API key cannot be empty")

    is_groq = api_key.startswith("gsk_")
    provider = "Groq" if is_groq else "OpenRouter"
    models_url = (
        "https://api.groq.com/openai/v1/models"
        if is_groq
        else "https://openrouter.ai/api/v1/key"
    )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

        if response.status_code == 200:
            return BaseValidationResponse(
                valid=True,
                message=f"{provider} API key is valid",
            )

        logger.debug(
            f"{provider} API key validation returned "
            f"{response.status_code}: {response.text[:300]}"
        )
        return BaseValidationResponse(
            valid=False,
            message=f"Invalid or unauthorized {provider} API key",
        )
    except Exception as e:
        logger.debug(f"{provider} API key validation failed: {e}")
        return BaseValidationResponse(
            valid=False,
            message=f"Could not validate {provider} API key",
        )

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
    client_id = data.api_key.strip()

    # Simkl's public trending endpoint returns data even for bogus client IDs,
    # so it cannot be used as credential validation. Check the documented
    # 64-character hexadecimal client-ID format here; OAuth performs the real
    # server-side verification of the registered application.
    if re.fullmatch(r"[0-9a-fA-F]{64}", client_id):
        return BaseValidationResponse(
            valid=True,
            message="Simkl client ID format is valid; OAuth confirms the registered app",
        )

    return BaseValidationResponse(
        valid=False,
        message="Simkl client ID must be a 64-character hexadecimal value",
    )
