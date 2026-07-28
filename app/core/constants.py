RECOMMENDATIONS_CATALOG_NAME: str = "Top Picks For You"
DEFAULT_MIN_ITEMS: int = 8
DEFAULT_CATALOG_LIMIT: int = 20

MAX_CATALOG_ITEMS: int = 100

DEFAULT_CONCURRENCY_LIMIT: int = 30

DEFAULT_MINIMUM_RATING_FOR_THEME_BASED_MOVIE: float = 7.2
DEFAULT_MINIMUM_RATING_FOR_THEME_BASED_TV: float = 6.8


# cache keys
LIBRARY_ITEMS_KEY: str = "watchly:library_items:{token}"
PROFILE_KEY: str = "watchly:profile:{token}:{content_type}"
WATCHED_SETS_KEY: str = "watchly:watched_sets:{token}:{content_type}"
CATALOG_KEY: str = "watchly:catalog:{token}:{type}:{id}"


DISCOVER_ONLY_EXTRA: list[dict] = [{"name": "genre", "isRequired": True, "options": ["All"], "optionsLimit": 1}]


DISCOVERY_SETTINGS: dict = {
    "mainstream": {
        "popularity.gte": 30,
        "vote_average.gte": 6.2,
        "vote_count.gte": 500,
    },
    "balanced": {
        "vote_average.gte": 6.0,
        "vote_count.gte": 50,
    },
    "gems": {
        "popularity.lte": 15,
        "vote_average.gte": 7.2,
        "vote_count.gte": 100,
    },
    # "all" is the permissive option, but previously demanded vote_count >= 100 --
    # stricter than "balanced" (50), which made it the narrowest setting of the four
    # for niche categories. The vote floor exists to filter unrated junk, not to
    # gate visibility, so it is set just high enough to do that job.
    "all": {
        "popularity.gte": 0,
        "vote_average.gte": 5.0,
        "vote_count.gte": 10,
    },
}
