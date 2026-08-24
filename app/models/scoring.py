from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class StremioState(BaseModel):
    """Represents the user state for a library item."""

    lastWatched: datetime | None = None
    timeWatched: int = 0
    timeOffset: int = 0
    overallTimeWatched: int = 0
    timesWatched: int = 0
    flaggedWatched: int = 0
    duration: int = 0
    video_id: str | None = None
    watched: str | None = None
    noNotif: bool = False
    season: int = 0
    episode: int = 0

    @field_validator("lastWatched", mode="before")
    @classmethod
    def parse_last_watched(cls, v):
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                return None
        return v


class StremioLibraryItem(BaseModel):
    """Represents a raw item from Stremio library."""

    id: str = Field(..., alias="_id")
    type: str
    name: str
    state: StremioState = Field(default_factory=StremioState)
    mtime: str = Field(default="", alias="_mtime")
    poster: str | None = None
    temp: bool
    removed: bool

    # Enriched fields (not in raw Stremio JSON, added by our service)
    is_loved: bool = Field(default=False, alias="_is_loved")
    is_liked: bool = Field(default=False, alias="_is_liked")
    interest_score: float = Field(default=0.0, alias="_interest_score")
    provider_status: str | None = Field(default=None, alias="_provider_status")

    class Config:
        populate_by_name = True


class ScoredItem(BaseModel):
    """
    A processed item with calculated scores.
    This is the output of the ScoringService.
    """

    item: StremioLibraryItem
    score: float
    completion_rate: float
    is_rewatched: bool
    is_recent: bool
    source_type: str  # 'loved' | 'watched' | 'liked'
