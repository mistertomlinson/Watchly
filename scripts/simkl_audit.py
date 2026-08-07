"""Live Simkl account audit helper.

Run inside the Watchly container after setting SIMKL_CLIENT_ID and providing a
short-lived SIMKL_ACCESS_TOKEN environment variable. Prints only counts and
redacted samples; it never prints credentials.
"""

from __future__ import annotations

import asyncio
import os
from collections import Counter

from app.services.simkl_provider import SimklApiClient, SimklLibraryProvider


async def main() -> None:
    client_id = os.getenv("SIMKL_CLIENT_ID", "").strip()
    access_token = os.getenv("SIMKL_ACCESS_TOKEN", "").strip()
    if not client_id or not access_token:
        raise SystemExit("SIMKL_CLIENT_ID and SIMKL_ACCESS_TOKEN are required")

    client = SimklApiClient(client_id, access_token)
    try:
        user = await client.get_user()
        library = await SimklLibraryProvider(client).get_library_items()
    finally:
        await client.close()

    account = user.get("account") if isinstance(user.get("account"), dict) else user
    display = account.get("username") or account.get("name") or account.get("id") or "unknown"

    print("===== COPY FROM HERE =====")
    print("SIMKL AUDIT")
    print("ACCOUNT:", display)
    for bucket in ("loved", "liked", "watched", "added", "disliked", "removed"):
        print(f"{bucket.upper()}: {len(library.get(bucket, []))}")

    watched = library.get("watched", [])
    print("WATCHED TYPES:", dict(Counter(item.get("type", "unknown") for item in watched)))
    print("WATCHED SOURCES:", dict(Counter(item.get("_source", "unknown") for item in watched)))
    print("RATING BUCKETS:", dict(Counter(item.get("_rating_bucket", "none") for item in watched)))

    print("SAMPLES:")
    for bucket in ("loved", "liked", "disliked"):
        sample = library.get(bucket, [])[:5]
        rendered = [f"{item.get('name')} ({item.get('_personal_rating')})" for item in sample]
        print(f"  {bucket}: {rendered}")
    print("===== STOP COPYING HERE =====")


if __name__ == "__main__":
    asyncio.run(main())
