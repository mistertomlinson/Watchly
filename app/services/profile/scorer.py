from typing import Any

from app.models.taste_profile import TasteProfile
from app.services.profile.constants import (
    GENRE_MAX_POSITIONS,
    GENRE_POSITION_WEIGHTS,
    RANKING_IGNORED_KEYWORD_IDS,
    RANKING_KEYWORD_MATCH_LIMIT,
    RANKING_WEIGHT_CAST,
    RANKING_WEIGHT_DIRECTOR,
    RANKING_WEIGHT_ERA,
    RANKING_WEIGHT_GENRE,
    RANKING_WEIGHT_KEYWORD,
)


class ProfileScorer:
    """
    Scores items against taste profile using unified function.
    """

    @staticmethod
    def score_item(item_metadata: dict[str, Any], profile: TasteProfile) -> float:
        """
        Score an item against the profile.

        Ranking deliberately uses a normalized 0-1 taste score. Broad metadata
        such as era/director/cast is secondary, while genre and meaningful
        thematic keyword matches carry most of the signal.
        """
        normalized = profile.normalize_for_ranking()

        genre_score = ProfileScorer._score_genres(item_metadata, normalized)
        keyword_score = ProfileScorer._score_keywords(item_metadata, normalized)
        cast_score = ProfileScorer._score_cast(item_metadata, normalized)
        director_score = ProfileScorer._score_directors(item_metadata, normalized)
        era_score = ProfileScorer._score_era(item_metadata, normalized)

        return (
            genre_score * RANKING_WEIGHT_GENRE
            + keyword_score * RANKING_WEIGHT_KEYWORD
            + cast_score * RANKING_WEIGHT_CAST
            + director_score * RANKING_WEIGHT_DIRECTOR
            + era_score * RANKING_WEIGHT_ERA
        )

    @staticmethod
    def _score_genres(item_metadata: dict[str, Any], normalized: dict[str, Any]) -> float:
        item_genres = item_metadata.get("genre_ids", []) or []
        if not item_genres:
            genres = item_metadata.get("genres", []) or []
            if isinstance(genres, list):
                item_genres = [
                    g.get("id")
                    for g in genres
                    if isinstance(g, dict) and g.get("id") is not None
                ]

        weighted_sum = 0.0
        weight_sum = 0.0
        for position, genre_id in enumerate(item_genres[:GENRE_MAX_POSITIONS]):
            weight = GENRE_POSITION_WEIGHTS[position]
            weighted_sum += float(normalized["genres"].get(genre_id, 0.0)) * weight
            weight_sum += weight
        return weighted_sum / weight_sum if weight_sum else 0.0

    @staticmethod
    def _score_keywords(item_metadata: dict[str, Any], normalized: dict[str, Any]) -> float:
        keyword_ids: list[int] = []

        raw_ids = item_metadata.get("keyword_ids", []) or []
        if isinstance(raw_ids, list):
            for keyword_id in raw_ids:
                try:
                    kid = int(keyword_id)
                except (TypeError, ValueError):
                    continue
                if kid not in RANKING_IGNORED_KEYWORD_IDS:
                    keyword_ids.append(kid)

        if not keyword_ids:
            keywords = item_metadata.get("keywords", {}) or {}
            if isinstance(keywords, dict):
                raw_keywords = keywords.get("keywords") or keywords.get("results") or []
            elif isinstance(keywords, list):
                raw_keywords = keywords
            else:
                raw_keywords = []

            for keyword in raw_keywords:
                if not isinstance(keyword, dict):
                    continue
                try:
                    kid = int(keyword.get("id"))
                except (TypeError, ValueError):
                    continue
                if kid in RANKING_IGNORED_KEYWORD_IDS:
                    continue
                keyword_ids.append(kid)

        scores = [float(normalized["keywords"].get(kid, 0.0)) for kid in keyword_ids]
        scores = sorted((score for score in scores if score > 0.0), reverse=True)
        strongest = scores[:RANKING_KEYWORD_MATCH_LIMIT]
        return sum(strongest) / len(strongest) if strongest else 0.0

    @staticmethod
    def _score_cast(item_metadata: dict[str, Any], normalized: dict[str, Any]) -> float:
        item_cast = ProfileScorer._extract_cast_ids(item_metadata)
        if not item_cast:
            return 0.0
        matches = [float(normalized["cast"].get(cast_id, 0.0)) for cast_id in item_cast]
        return sum(matches) / len(matches) if matches else 0.0

    @staticmethod
    def _score_directors(item_metadata: dict[str, Any], normalized: dict[str, Any]) -> float:
        item_directors = ProfileScorer._extract_director_ids(item_metadata)
        if not item_directors:
            return 0.0
        matches = [float(normalized["directors"].get(director_id, 0.0)) for director_id in item_directors]
        return sum(matches) / len(matches) if matches else 0.0

    @staticmethod
    def _score_era(item_metadata: dict[str, Any], normalized: dict[str, Any]) -> float:
        year = item_metadata.get("release_date") or item_metadata.get("first_air_date") or item_metadata.get("released")
        if not year:
            return 0.0
        try:
            year_int = int(str(year)[:4])
        except (ValueError, TypeError):
            return 0.0
        era = ProfileScorer._year_to_era(year_int)
        return float(normalized["eras"].get(era, 0.0))

    @staticmethod
    def _extract_cast_ids(item_metadata: dict[str, Any]) -> list[int]:
        """Extract unique top-5 cast IDs from item metadata."""
        cast_ids = []
        seen = set()
        credits = item_metadata.get("credits", {}) or {}
        cast_list = credits.get("cast", []) or []
        for actor in cast_list[:5]:  # Top 5 only
            if isinstance(actor, dict):
                actor_id = actor.get("id")
                if actor_id and actor_id not in seen:
                    cast_ids.append(actor_id)
                    seen.add(actor_id)
        return cast_ids

    @staticmethod
    def _extract_director_ids(item_metadata: dict[str, Any]) -> list[int]:
        """Extract unique director/creator IDs from item metadata."""
        director_ids = []
        seen = set()
        credits = item_metadata.get("credits", {}) or {}
        crew_list = credits.get("crew", []) or []
        for crew_member in crew_list:
            if not isinstance(crew_member, dict):
                continue
            job = str(crew_member.get("job") or "").lower()
            if job not in {"director", "creator"}:
                continue
            director_id = crew_member.get("id")
            if director_id and director_id not in seen:
                director_ids.append(director_id)
                seen.add(director_id)
        return director_ids

    @staticmethod
    def _extract_country_codes(item_metadata: dict[str, Any]) -> list[str]:
        """Extract country codes from item metadata."""
        countries = []
        production_countries = item_metadata.get("production_countries", []) or []
        for country in production_countries:
            if isinstance(country, dict):
                country_code = country.get("iso_3166_1")
            else:
                country_code = country
            if country_code:
                countries.append(country_code)
        return countries

    @staticmethod
    def _year_to_era(year: int) -> str:
        """Convert year to era bucket."""
        if year < 1970:
            return "pre-1970s"
        if year < 1980:
            return "1970s"
        if year < 1990:
            return "1980s"
        if year < 2000:
            return "1990s"
        if year < 2010:
            return "2000s"
        if year < 2020:
            return "2010s"
        return "2020s"
