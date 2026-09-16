"""The Odds API adapter - the project's live price source.

Why this provider: it redistributes genuine **betting exchange** prices
(Betfair Exchange, Smarkets, Matchbook) alongside conventional bookmakers, and
needs only an email to obtain a key. Exchange prices carry no bookmaker margin,
which is the whole premise of the model. Several exchanges arrive in the same
response, giving per-fixture fallback when one is thin, and Pinnacle is
available in the EU region as a low-margin bookmaker backstop.

Measured on live closing prices: exchange overround 1.0047 on the back side and
0.9926 on the lay side, against 1.04-1.08 for conventional bookmakers.

### Credit budget

A request costs ``markets x regions``, independent of how many bookmakers come
back. Asking for ``h2h`` and ``totals`` across the ``uk`` region is therefore
**2 credits**, and returns three exchanges at once. On the free tier's 500
credits a month that is ample for a weekly tool, but responses are cached to
disk anyway - re-running a notebook should never burn quota.

### What the exchange does and does not supply

**Match odds** arrive with both sides: ``h2h`` is the back side and ``h2h_lay``
the lay side, so a fair price is taken as the midpoint in probability space -
the same treatment a full order book would get. Using the back side alone would
overstate every probability by roughly half a percent before normalisation.

**Over/under is not published by the exchange at all.** Totals therefore come
from a consensus of de-vigged bookmaker lines. That second market is not
optional: without it the scoreline fit has only the three 1X2 outcomes for three
free parameters, which is exactly determined - the goal *total* ends up pinned
by the rho prior rather than by any data, and the residual carries no
information. Adding totals moved one fixture's expected goals from 2.48 to 3.27.

There is no order-book depth or matched-volume figure here, so the liquidity
gating a direct exchange feed allows is not possible; cross-source agreement is
the available substitute.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..config import CACHE_DIR, credentials, params
from ..model.scoreline import ScorelineDistribution, fit_to_markets
from ..naming import match_fixtures, normalise, team_key

BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"

# Preference order: true exchanges first, then the sharpest bookmaker.
EXCHANGE_PREFERENCE = ["betfair_ex_uk", "betfair_ex_eu", "smarkets", "matchbook", "pinnacle"]


class OddsAPIError(RuntimeError):
    pass


class OddsAPIClient:
    """Read-only client with disk caching and quota reporting."""

    def __init__(self, api_key: str | None = None, cache_dir: Path | None = None):
        creds = credentials()
        self.api_key = api_key or creds.get("odds_api_key")
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.requests_remaining: int | None = None
        self.requests_used: int | None = None
        self.last_cost: int | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def get_odds(
        self,
        regions: str = "uk",
        markets: str = "h2h,totals",
        bookmakers: str | None = None,
        max_age: float = 900.0,
        sport: str = SPORT,
    ) -> list[dict]:
        """Upcoming fixtures with odds.

        Cached for ``max_age`` seconds; pass ``max_age=0`` to force a refresh.
        Every cache hit is a credit saved.
        """
        if not self.api_key:
            raise OddsAPIError(
                "ODDS_API_KEY is not set. Get a free key at https://the-odds-api.com/ "
                "and put it in .env (see .env.example)."
            )
        tag = f"oddsapi_{sport}_{regions}_{markets.replace(',', '-')}"
        if bookmakers:
            tag += "_" + bookmakers.replace(",", "-")
        cache_file = self.cache_dir / f"{tag}.json"

        if cache_file.exists() and max_age > 0:
            if time.time() - cache_file.stat().st_mtime < max_age:
                with open(cache_file, encoding="utf-8") as fh:
                    payload = json.load(fh)
                self.last_cost = 0
                return payload["data"]

        query: dict[str, Any] = {
            "apiKey": self.api_key,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        # `bookmakers` replaces `regions` when given, and is the cheaper filter.
        if bookmakers:
            query["bookmakers"] = bookmakers
        else:
            query["regions"] = regions

        resp = self.session.get(f"{BASE}/sports/{sport}/odds", params=query, timeout=40)
        if resp.status_code == 401:
            raise OddsAPIError("API key rejected (401) - check ODDS_API_KEY")
        if resp.status_code == 422:
            raise OddsAPIError(f"invalid request (422): {resp.text[:200]}")
        resp.raise_for_status()

        # Quota lives in the response headers, not the body.
        def _hdr(name):
            v = resp.headers.get(name)
            return int(float(v)) if v not in (None, "") else None

        self.requests_remaining = _hdr("x-requests-remaining")
        self.requests_used = _hdr("x-requests-used")
        self.last_cost = _hdr("x-requests-last")

        data = resp.json()
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump({"fetched_at": time.time(), "data": data}, fh, ensure_ascii=False)
        return data

    def quota(self) -> str:
        if self.requests_remaining is None:
            return "quota unknown (no live call yet this session)"
        return (
            f"{self.requests_remaining} credits remaining "
            f"({self.requests_used} used, last call cost {self.last_cost})"
        )


# --------------------------------------------------------------------------
# Odds -> probabilities -> scoreline
# --------------------------------------------------------------------------
def _pick_bookmaker(event: dict, preference: list[str], needs: str = "h2h") -> dict | None:
    by_key = {b.get("key"): b for b in event.get("bookmakers", [])}
    for key in preference:
        b = by_key.get(key)
        if b and any(m.get("key") == needs for m in b.get("markets", [])):
            return b
    for b in by_key.values():
        if any(m.get("key") == needs for m in b.get("markets", [])):
            return b
    return None


def _two_sided_h2h(book: dict) -> tuple[dict[str, float], str]:
    """Fair 1X2 probabilities, using the lay side when the exchange offers it.

    An exchange quotes two prices per runner. ``h2h`` is the back side and
    ``h2h_lay`` the lay side, and the fair value sits between them - so where
    both exist the midpoint is taken **in probability space**, not odds space,
    because 1/x is convex and averaging odds then inverting is biased.

    Observed on live data: back overround 1.0047, lay 0.9926. Using the back
    side alone therefore overstates every probability by roughly half a percent
    before normalisation.
    """
    markets = {m.get("key"): m for m in book.get("markets", [])}
    back = markets.get("h2h")
    lay = markets.get("h2h_lay")
    if back is None:
        return {}, "none"

    back_p = {o["name"]: 1.0 / o["price"] for o in back.get("outcomes", [])
              if o.get("price", 0) > 1}
    if lay is None:
        total = sum(back_p.values())
        return ({k: v / total for k, v in back_p.items()} if total else {}), "back-only"

    lay_p = {o["name"]: 1.0 / o["price"] for o in lay.get("outcomes", [])
             if o.get("price", 0) > 1}
    mid = {k: (v + lay_p[k]) / 2 if k in lay_p else v for k, v in back_p.items()}
    total = sum(mid.values())
    return ({k: v / total for k, v in mid.items()} if total else {}), "back+lay mid"


def _totals_consensus(event: dict) -> tuple[dict[float, float], int, list[str]]:
    """Consensus P(total goals > line), de-vigged per source then medianed.

    The exchanges do not publish totals through this API, so this market
    has to come from elsewhere. Each source's two-runner book is normalised
    individually - which removes a bookmaker's margin cleanly, since with two
    outcomes the margin is close to symmetric - and the median across sources is
    taken to blunt any single book's bias.

    Without this the scoreline fit has only the three 1X2 outcomes for three
    free parameters: exactly determined, zero residual by construction, and the
    goal *total* pinned by the rho prior rather than by any data.
    """
    import statistics

    per_line: dict[float, list[float]] = {}
    sources: list[str] = []
    for b in event.get("bookmakers", []):
        for m in b.get("markets", []):
            if m.get("key") != "totals":
                continue
            by_line: dict[float, dict[str, float]] = {}
            for o in m.get("outcomes", []):
                pt, price = o.get("point"), o.get("price")
                if pt is None or not price or price <= 1:
                    continue
                by_line.setdefault(float(pt), {})[normalise(o["name"])] = 1.0 / price
            for line, side in by_line.items():
                if "over" in side and "under" in side:
                    tot = side["over"] + side["under"]
                    if tot > 0:
                        per_line.setdefault(line, []).append(side["over"] / tot)
                        if b["key"] not in sources:
                            sources.append(b["key"])
    return ({line: statistics.median(v) for line, v in per_line.items()},
            max((len(v) for v in per_line.values()), default=0), sources)


def scoreline_from_event(
    event: dict, home_name: str, away_name: str, preference: list[str] | None = None
) -> tuple[ScorelineDistribution | None, str]:
    """Fit a scoreline distribution from one event's odds.

    Returns ``(distribution, note)``; the note explains any failure.
    """
    preference = preference or EXCHANGE_PREFERENCE
    book = _pick_bookmaker(event, preference, needs="h2h")
    if book is None:
        return None, "no bookmaker offering h2h"

    h_keys, a_keys = team_key(home_name), team_key(away_name)
    probs, side = _two_sided_h2h(book)
    p_home = p_draw = p_away = None
    for name, p in probs.items():
        if normalise(name) == "draw":
            p_draw = p
        elif team_key(name) & h_keys:
            p_home = p
        elif team_key(name) & a_keys:
            p_away = p

    used: list[str] = []
    if None not in (p_home, p_draw, p_away):
        used.append("h2h")

    # Totals come from whichever sources offer them - the exchange does not.
    over_lines, n_sources, tot_sources = _totals_consensus(event)
    if over_lines:
        used.append(f"totals({n_sources})")

    n_obs = sum(x is not None for x in (p_home, p_draw, p_away)) + len(over_lines)
    if n_obs < 2:
        return None, f"only {n_obs} usable observation(s) from {book.get('key')}"

    try:
        dist = fit_to_markets(
            p_home=p_home, p_draw=p_draw, p_away=p_away,
            over_lines=over_lines or None,
            source=f"oddsapi:{book.get('key')}:" + "+".join(used),
            markets_used=used,
        )
    except ValueError as exc:
        return None, f"fit failed: {exc}"

    dist.diagnostics["_meta"] = {
        "h2h_source": book.get("key"), "h2h_side": side,
        "totals_sources": tot_sources, "n_observations": n_obs,
    }
    note = f"{book.get('key')} {side}, {n_obs} obs"
    if not over_lines:
        # Worth flagging loudly: with only the three 1X2 outcomes the fit is
        # exactly determined, so the total-goals estimate is not identified by
        # data and the residual carries no information.
        note += " - NO TOTALS, fit is exactly determined"
    return dist, note


def scorelines_for_gameweek(
    fixtures_gw: pd.DataFrame,
    client: OddsAPIClient | None = None,
    preference: list[str] | None = None,
    max_age: float = 900.0,
) -> tuple[dict[int, ScorelineDistribution], list[str]]:
    """Scoreline distributions for a gameweek's fixtures.

    Returns ``(fixture_id -> distribution, notes)``. Fixtures that cannot be
    priced are simply absent, so the caller's fallback model fills them in.
    """
    client = client or OddsAPIClient()
    cfg = params().get("oddsapi", {})
    regions = cfg.get("regions", "uk")
    markets = cfg.get("markets", "h2h,totals")
    preference = preference or cfg.get("bookmaker_preference", EXCHANGE_PREFERENCE)

    events = client.get_odds(regions=regions, markets=markets, max_age=max_age)
    notes: list[str] = []

    mapping, unmatched = match_fixtures(fixtures_gw, events)
    for u in unmatched:
        notes.append(f"no odds event matched: {u}")

    by_id = {str(e.get("id")): e for e in events}
    out: dict[int, ScorelineDistribution] = {}
    for fid, ev_id in mapping.items():
        event = by_id.get(ev_id)
        if event is None:
            continue
        fx = fixtures_gw[fixtures_gw["fixture_id"] == fid].iloc[0]
        dist, note = scoreline_from_event(
            event, fx.get("team_h_name", fx["team_h_short"]),
            fx.get("team_a_name", fx["team_a_short"]), preference,
        )
        if dist is not None:
            out[fid] = dist
        else:
            notes.append(f"{fx['team_h_short']} v {fx['team_a_short']}: {note}")
    return out, notes


def compare_sources(fixtures_gw: pd.DataFrame, client: OddsAPIClient | None = None,
                    max_age: float = 900.0) -> pd.DataFrame:
    """Price every fixture from each available exchange, side by side.

    Useful as a sanity check: independent exchanges should agree closely on a
    liquid market. A fixture where they diverge is one where at least one book
    is thin or stale, and the projection deserves less confidence.
    """
    client = client or OddsAPIClient()
    cfg = params().get("oddsapi", {})
    events = client.get_odds(regions=cfg.get("regions", "uk"),
                             markets=cfg.get("markets", "h2h,totals"), max_age=max_age)
    mapping, _ = match_fixtures(fixtures_gw, events)
    by_id = {str(e.get("id")): e for e in events}

    rows = []
    for fid, ev_id in mapping.items():
        event = by_id.get(ev_id)
        fx = fixtures_gw[fixtures_gw["fixture_id"] == fid].iloc[0]
        for key in EXCHANGE_PREFERENCE:
            if not any(b.get("key") == key for b in event.get("bookmakers", [])):
                continue
            dist, _ = scoreline_from_event(
                event, fx.get("team_h_name", fx["team_h_short"]),
                fx.get("team_a_name", fx["team_a_short"]), preference=[key],
            )
            if dist is None:
                continue
            rows.append({
                "fixture": f"{fx['team_h_short']} v {fx['team_a_short']}",
                "source": key,
                "xG_home": round(dist.lambda_home, 3),
                "xG_away": round(dist.lambda_away, 3),
                "P(home)": round(dist.p_win(True), 4),
                "CS_home": round(dist.p_clean_sheet(True), 4),
                "CS_away": round(dist.p_clean_sheet(False), 4),
                "residual": round(dist.fit_residual, 5),
            })
    return pd.DataFrame(rows)
