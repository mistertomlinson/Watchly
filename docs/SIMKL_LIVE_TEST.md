# Simkl live-test checklist

Target branch: `agent/simkl-full-provider`

## Required server configuration

Register the Simkl application callback as:

`https://watchlyselfhost.duckdns.org/tokens/simkl/callback`

Set these only in the compose `.env`; never commit them:

- `SIMKL_CLIENT_ID`
- `SIMKL_CLIENT_SECRET`

## First deployment order

1. Deploy profile 1 only.
2. Rebuild Watchly and verify container startup.
3. Confirm `GET /tokens/simkl/config` reports `configured: true`.
4. Open the configure page and connect Simkl.
5. Complete OAuth and create the addon manifest.
6. Inspect container logs for Simkl library counts and profile-build errors.
7. Run the live audit helper with a temporary access token.
8. Verify rating buckets and generated rows before touching profile 2.

## Required behavior

- 10: loved; strongest taste influence.
- 7-9: liked; lighter taste influence.
- 3-6: neutral; usable for Because You Watched and watched exclusions only.
- 1-2: disliked; never recommendation anchors.
- Providers are selected independently; no Trakt/Simkl merging.

## Known blocker before production

`SmartSampler` currently includes neutral rated watched/added items in general profile sampling. The regression test in `tests/test_neutral_rating_sampling.py` documents the required fix. Patch the sampler so an item with `_personal_rating` from 3 through 6 is omitted from general profile sampling, while unrated watched items retain existing behavior.

## Live data checks

Copy back only the delimited output from `scripts/simkl_audit.py`. Confirm:

- account identity is correct;
- loved/liked/disliked counts approximately match Simkl;
- ratings 3-6 appear in watched but not loved/liked/disliked;
- ratings 1-2 appear only in disliked and watched as appropriate;
- IMDb/TMDB-backed IDs dominate; investigate unresolved `simkl:` IDs;
- movie/series totals are plausible;
- anime movies/specials are not badly misclassified.

## Cache checks

After successful setup:

1. Load the manifest and a sample of every enabled catalog.
2. Record library/profile counts.
3. Clear only the Simkl user's library/profile/manifest cache keys.
4. Reload the manifest.
5. Confirm the library is fetched from Simkl again and catalogs rebuild successfully.
6. Confirm the other Watchly profile remains untouched, including Redis DB separation for p2 (`-n 1`).

## Do not merge until

- Docker starts cleanly;
- provider and Simkl tests pass;
- OAuth succeeds against the registered app;
- a real account imports expected counts;
- neutral-rating sampling is fixed;
- representative movie and series catalogs return results;
- cache-expiry refresh is verified;
- profile 2 is tested separately using Redis DB 1.
