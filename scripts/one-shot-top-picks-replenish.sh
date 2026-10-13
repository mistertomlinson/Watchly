#!/usr/bin/env bash
set -euo pipefail

LOG=/tmp/watchly-top-picks-replenish.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
trap 'sudo cp "$LOG" /var/www/seasonal-output/latest-output.txt >/dev/null 2>&1 || true' EXIT

echo "===== TOP PICKS DEDUPE REPLENISHMENT ====="
cd /home/ubuntu/watchly

echo "===== VERIFY CLEAN TRACKED WORKTREE ====="
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "STOP: tracked local changes exist"
  git status --short
  exit 1
fi

echo "===== UPDATE SOURCE ====="
git fetch origin
git switch feature/theme-rotation-v2
git pull --ff-only origin feature/theme-rotation-v2

if grep -q 'TOP_PICKS_DEDUPE_RESERVE = 10' app/services/recommendation/catalog_service.py && grep -q 'qualified Top Picks reserve' app/services/manifest.py; then
  echo "Patch already present; skipping source edit"
else
python3 - <<'PY'

from pathlib import Path

def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one match, found {count}")
    p.write_text(text.replace(old, new, 1))

# 1) Top Picks: keep the normal 20-item display selection unchanged, then append
# an already-scored quality-gated reserve for manifest dedupe replenishment.
path = "app/services/recommendation/top_picks_v2.py"
replace_once(
    path,
    '''        gemini_api_key: str | None = None,
        library_items_raw: dict | None = None,
    ) -> list[dict[str, Any]]:
''',
    '''        gemini_api_key: str | None = None,
        library_items_raw: dict | None = None,
        reserve: int = 0,
    ) -> list[dict[str, Any]]:
''',
)
replace_once(
    path,
    '''        target = max(1, min(int(limit), MAX_CATALOG_ITEMS))
        candidate_target = max(self.MIN_CANDIDATE_HEADROOM, target * 3)
''',
    '''        target = max(1, min(int(limit), MAX_CATALOG_ITEMS))
        reserve_target = max(0, min(int(reserve), MAX_CATALOG_ITEMS - target))
        candidate_target = max(self.MIN_CANDIDATE_HEADROOM, target * 3)
''',
)
replace_once(
    path,
    '''        rotated = DailyRotation.rotate_items(qualified, rotation_seed)
        result = rotated[:target]

        elapsed = time.time() - start_time
        logger.info(
            f"Top picks complete: returned={len(result)}, qualified={len(qualified)}, "
            f"scored={len(scored_candidates)}, enriched={len(enriched)}, "
            f"raw_sources={dict(source_counts)}, elapsed={elapsed:.2f}s"
        )
        return result
''',
    '''        rotated = DailyRotation.rotate_items(qualified, rotation_seed)
        display_result = rotated[:target]

        # Keep a hidden reserve behind the display selection for coordinated
        # manifest dedupe. These candidates have already survived watched,
        # settings, metadata, home-availability, and baseline quality gates.
        reserve_items: list[dict[str, Any]] = []
        if reserve_target:
            selected_ids = {
                self._canonical_tmdb_id(item) or item.get("id")
                for item in display_result
            }
            min_rating, min_votes = RecommendationFiltering.get_quality_thresholds(
                self.user_settings
            )
            for _, item in scored_candidates:
                if len(reserve_items) >= reserve_target:
                    break
                identity = self._canonical_tmdb_id(item) or item.get("id")
                if not identity or identity in selected_ids:
                    continue
                vote_count = int(item.get("vote_count") or 0)
                vote_avg = float(item.get("vote_average") or 0.0)
                weighted = RecommendationScoring.weighted_rating(
                    vote_avg,
                    vote_count,
                    C=7.2 if mtype == "tv" else 6.8,
                )
                if vote_count < min_votes or weighted < min_rating:
                    continue
                reserve_items.append(item)
                selected_ids.add(identity)

        result = display_result + reserve_items

        elapsed = time.time() - start_time
        logger.info(
            f"Top picks complete: returned={len(display_result)}, reserve={len(reserve_items)}, "
            f"qualified={len(qualified)}, scored={len(scored_candidates)}, "
            f"enriched={len(enriched)}, raw_sources={dict(source_counts)}, "
            f"elapsed={elapsed:.2f}s"
        )
        return result
''',
)

# 2) Catalog staging: request a reserve only for hidden staged Top Picks builds.
# Ordinary direct/published builds still produce exactly the normal display size.
path = "app/services/recommendation/catalog_service.py"
replace_once(
    path,
    '''BACKGROUND_CATALOG_REFRESH_LIMIT = 2
''',
    '''BACKGROUND_CATALOG_REFRESH_LIMIT = 2
TOP_PICKS_DEDUPE_RESERVE = 10
''',
)
replace_once(
    path,
    '''        limit: int,
        user_settings: UserSettings | None = None,
    ) -> list[dict[str, Any]]:
''',
    '''        limit: int,
        user_settings: UserSettings | None = None,
        top_picks_reserve: int = 0,
    ) -> list[dict[str, Any]]:
''',
)
replace_once(
    path,
    '''                    gemini_api_key=gemini_key,
                    library_items_raw=library_items,
                )
''',
    '''                    gemini_api_key=gemini_key,
                    library_items_raw=library_items,
                    reserve=top_picks_reserve,
                )
''',
)
replace_once(
    path,
    '''                limit=DEFAULT_CATALOG_LIMIT,
                user_settings=user_settings,
            )
''',
    '''                limit=DEFAULT_CATALOG_LIMIT,
                user_settings=user_settings,
                top_picks_reserve=(
                    TOP_PICKS_DEDUPE_RESERVE
                    if stage_only and catalog_id == "watchly.rec"
                    else 0
                ),
            )
''',
)
replace_once(
    path,
    '''            cleaned = shuffle_data_if_needed(user_settings, catalog_id, cleaned)

            data = {"metas": cleaned}
''',
    '''            if (
                stage_only
                and catalog_id == "watchly.rec"
                and len(cleaned) > DEFAULT_CATALOG_LIMIT
            ):
                # Shuffle only the normal display selection. The appended reserve
                # remains behind it so dedupe can pull replacements without
                # displacing valid primary picks unless a hole actually opens.
                primary = shuffle_data_if_needed(
                    user_settings,
                    catalog_id,
                    cleaned[:DEFAULT_CATALOG_LIMIT],
                )
                cleaned = primary + cleaned[DEFAULT_CATALOG_LIMIT:]
            else:
                cleaned = shuffle_data_if_needed(user_settings, catalog_id, cleaned)

            data = {"metas": cleaned}
''',
)

# 3) Manifest dedupe: use the staged Top Picks reserve to refill holes and
# publish exactly DEFAULT_CATALOG_LIMIT when enough valid unique picks exist.
path = "app/services/manifest.py"
replace_once(
    path,
    '''from app.core.config import settings
from app.core.security import redact_token
''',
    '''from app.core.config import settings
from app.core.constants import DEFAULT_CATALOG_LIMIT
from app.core.security import redact_token
''',
)
replace_once(
    path,
    '''            if not duplicates:
                continue

            retained: list[set[str]] = [set() for _ in rows]
''',
    '''            retained: list[set[str]] = [set() for _ in rows]
''',
)
replace_once(
    path,
    '''                removed = len(row["metas"]) - len(new_metas)

                # Always rewrite the isolated staged snapshot from its undeduped source.
''',
    '''                deduped_count = len(new_metas)
                replenished = 0

                if row["catalog_id"] == "watchly.rec":
                    primary_ids = {
                        self._catalog_meta_identity(meta)
                        for meta in row["metas"][:DEFAULT_CATALOG_LIMIT]
                    }
                    primary_ids.discard(None)

                    # The raw Top Picks snapshot contains the normal display picks
                    # followed by quality-gated reserves. After cross-row ownership
                    # removes duplicates, preserve order and let reserve entries fill
                    # any holes before publishing the standard display size.
                    if len(new_metas) > DEFAULT_CATALOG_LIMIT:
                        new_metas = new_metas[:DEFAULT_CATALOG_LIMIT]

                    replenished = sum(
                        1
                        for meta in new_metas
                        if self._catalog_meta_identity(meta) not in primary_ids
                    )
                    if replenished:
                        logger.info(
                            f"[{redact_token(token)}...] [Dedupe] {content_type} "
                            f"'{row['name']}' replenished {replenished} placement(s) "
                            f"from qualified Top Picks reserve; final={len(new_metas)}"
                        )
                    if (
                        len(row["metas"]) >= DEFAULT_CATALOG_LIMIT
                        and len(new_metas) < DEFAULT_CATALOG_LIMIT
                    ):
                        logger.warning(
                            f"[{redact_token(token)}...] [Dedupe] {content_type} "
                            f"'{row['name']}' reserve exhausted at {len(new_metas)}/"
                            f"{DEFAULT_CATALOG_LIMIT}"
                        )

                removed = len(row["metas"]) - deduped_count

                # Always rewrite the isolated staged snapshot from its undeduped source.
''',
)
replace_once(
    path,
    '''                        f"'{row['name']}' {len(row['metas'])} -> {len(new_metas)} "
                        f"(removed {removed} duplicate placements)"
''',
    '''                        f"'{row['name']}' {len(row['metas'])} -> {deduped_count} "
                        f"(removed {removed} duplicate placements)"
''',
)

test_path = Path("tests/test_manifest_top_picks_replenishment.py")
test_path.write_text('''from unittest.mock import AsyncMock, patch

import pytest

from app.services import manifest as manifest_module
from app.services.manifest import ManifestService


def _meta(media_id: str) -> dict:
    return {"id": media_id, "type": "movie", "name": media_id}


@pytest.mark.asyncio
async def test_top_picks_replenishes_from_hidden_reserve_after_cross_row_dedupe():
    service = ManifestService()

    primary = [_meta(f"tt{index:07d}") for index in range(1, 21)]
    reserve = [_meta(f"tt{index:07d}") for index in range(21, 31)]

    # The sibling row is deliberately smaller before duplicate allocation, so
    # the water-filling deduper awards the two shared primary titles away from
    # Top Picks. Its reserve must then refill those holes back to 20.
    sibling = [
        _meta("tt0000001"),
        _meta("tt0000002"),
        *[_meta(f"tt{index:07d}") for index in range(101, 119)],
    ]

    staged = {
        ("movie", "watchly.rec"): ({"metas": primary + reserve}, 1, 7),
        ("movie", "watchly.theme.test"): ({"metas": sibling}, 1, 7),
    }
    rewritten = {}

    async def get_staged(_token, content_type, catalog_id):
        return staged.get((content_type, catalog_id))

    async def get_staged_raw(_token, content_type, catalog_id):
        return staged.get((content_type, catalog_id))

    async def set_staged(_token, content_type, catalog_id, data, source_revision=None):
        rewritten[(content_type, catalog_id)] = data

    manifest = {
        "catalogs": [
            {"type": "movie", "id": "watchly.rec", "name": "Top Movies for You"},
            {"type": "movie", "id": "watchly.theme.test", "name": "Theme"},
        ]
    }

    with (
        patch.object(
            manifest_module.user_cache,
            "get_staged_catalog",
            new=AsyncMock(side_effect=get_staged),
        ),
        patch.object(
            manifest_module.user_cache,
            "get_staged_raw_catalog",
            new=AsyncMock(side_effect=get_staged_raw),
        ),
        patch.object(
            manifest_module.user_cache,
            "set_staged_catalog",
            new=AsyncMock(side_effect=set_staged),
        ),
    ):
        await service._dedupe_prewarmed_catalogs("token", manifest)

    top = rewritten[("movie", "watchly.rec")]["metas"]
    theme = rewritten[("movie", "watchly.theme.test")]["metas"]

    assert len(top) == 20
    assert {m["id"] for m in top}.isdisjoint({m["id"] for m in theme})
    assert any(m["id"] in {"tt0000021", "tt0000022"} for m in top)
''')

print("PATCH_APPLIED")

PY
fi

echo "===== STATIC CHECK ====="
python3 -m py_compile app/services/recommendation/top_picks_v2.py app/services/recommendation/catalog_service.py app/services/manifest.py tests/test_manifest_top_picks_replenishment.py
git diff --check

echo "===== TEST ====="
docker inspect watchly-p1-watchly-1 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep '=' > /tmp/watchly-test.env
docker run --rm --entrypoint sh --network watchly-p1_default --env-file /tmp/watchly-test.env -e UV_PROJECT_ENVIRONMENT=/tmp/v -e PYTHONPATH=/workspace -v /home/ubuntu/watchly:/workspace:ro -w /workspace watchly-p1 -c 'uv run pytest -q tests/test_manifest_top_picks_replenishment.py tests/test_top_picks_pipeline_v2.py tests/test_catalog_build_controls.py tests/test_recommendation_home_availability_global.py tests/test_top_picks_home_availability.py'

echo "===== COMMIT AND PUSH ====="
rm -f scripts/one-shot-top-picks-replenish.sh
git add app/services/recommendation/top_picks_v2.py app/services/recommendation/catalog_service.py app/services/manifest.py tests/test_manifest_top_picks_replenishment.py scripts/one-shot-top-picks-replenish.sh
if git diff --cached --quiet; then
  echo "No new source changes to commit"
else
  git commit -m "Replenish Top Picks after cross-row dedupe"
  git push origin feature/theme-rotation-v2
fi

echo "===== SAVE ROLLBACK IMAGES ====="
STAMP=$(date +%Y%m%d-%H%M%S)
docker image inspect watchly-p1 >/dev/null 2>&1 && docker tag watchly-p1 "watchly-p1:pre-top-picks-replenish-$STAMP" || true
docker image inspect watchly-p2 >/dev/null 2>&1 && docker tag watchly-p2 "watchly-p2:pre-top-picks-replenish-$STAMP" || true

echo "===== BUILD P1 ====="
cd /home/ubuntu/watchly-p1
docker compose build watchly

echo "===== BUILD P2 ====="
cd /home/ubuntu/watchly-p2
docker compose build watchly

echo "===== STOP APP CONTAINERS ====="
cd /home/ubuntu/watchly-p1
docker compose stop watchly
cd /home/ubuntu/watchly-p2
docker compose stop watchly

echo "===== CLEAR ONLY PREPARED/STAGED GENERATIONS ====="
docker exec watchly-p1-redis-1 sh -c 'redis-cli --scan --pattern "watchly:manifest:prepared:*" | xargs -r redis-cli DEL >/dev/null; redis-cli --scan --pattern "watchly:catalog_staged:*" | xargs -r redis-cli DEL >/dev/null; redis-cli --scan --pattern "watchly:catalog_staged_raw:*" | xargs -r redis-cli DEL >/dev/null'
docker exec watchly-p2-redis-1 sh -c 'redis-cli --scan --pattern "watchly:manifest:prepared:*" | xargs -r redis-cli DEL >/dev/null; redis-cli --scan --pattern "watchly:catalog_staged:*" | xargs -r redis-cli DEL >/dev/null; redis-cli --scan --pattern "watchly:catalog_staged_raw:*" | xargs -r redis-cli DEL >/dev/null'

echo "===== START NEW P1/P2 ====="
cd /home/ubuntu/watchly-p1
docker compose up -d --no-deps --force-recreate watchly
cd /home/ubuntu/watchly-p2
docker compose up -d --no-deps --force-recreate watchly
sleep 5

echo "===== VERIFY LIVE CODE ====="
for C in watchly-p1-watchly-1 watchly-p2-watchly-1; do
  echo "--- $C ---"
  docker exec "$C" grep -n 'TOP_PICKS_DEDUPE_RESERVE = 10' /app/app/services/recommendation/catalog_service.py
  docker exec "$C" grep -n 'reserve: int = 0' /app/app/services/recommendation/top_picks_v2.py
  docker exec "$C" grep -n 'qualified Top Picks reserve' /app/app/services/manifest.py
  docker inspect "$C" --format 'status={{.State.Status}} restarts={{.RestartCount}} image={{.Image}}'
done

echo "===== FINAL ====="
echo "HEAD=$(git -C /home/ubuntu/watchly rev-parse HEAD)"
git -C /home/ubuntu/watchly status --short
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | grep -E 'NAMES|watchly-p[12]'
echo "Active published catalogs were preserved; no manifest promotion was forced."
echo "===== STOP COPYING HERE ====="
