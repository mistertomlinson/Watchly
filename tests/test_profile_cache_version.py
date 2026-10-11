from app.services.user_cache import UserCacheService


def test_profile_cache_key_is_versioned():
    assert UserCacheService._profile_key("token123", "movie") == "watchly:profile:v2:token123:movie"
    assert UserCacheService._profile_key("token123", "series") == "watchly:profile:v2:token123:series"
