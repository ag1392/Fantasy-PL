"""Verify The Odds API setup, end to end.

Staged so that a failure tells you which step broke rather than returning a
generic error. Costs **2 credits** in total (one call for h2h + totals), and
reports your remaining balance.

    python scripts/check_odds_api.py
    python scripts/check_odds_api.py --event 6
    python scripts/check_odds_api.py --fresh     # bypass the response cache

Stages:
    0  .env present, key set, gitignored
    1  the API accepts the key, and quota is reported
    2  Premier League fixtures come back, with which bookmakers
    3  FPL fixtures join onto the odds events
    4  scorelines fit from the prices
    5  independent exchanges agree with each other
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from fplfh.config import ROOT, credentials, params
from fplfh.fpl import FPLClient, build_fixtures
from fplfh.naming import match_fixtures
from fplfh.odds import EXCHANGE_PREFERENCE, OddsAPIClient, compare_sources
from fplfh.odds.oddsapi import OddsAPIError, scoreline_from_event

OK, BAD, WARN = "[ok]", "[FAIL]", "[warn]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, default=None, help="gameweek (default: next)")
    ap.add_argument("--fresh", action="store_true", help="ignore the cache")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 72)
    print("The Odds API setup check")
    print("=" * 72)

    # ------------------------------------------------------------ stage 0
    print("\nStage 0 - configuration")
    env = ROOT / ".env"
    if not env.exists():
        print(f"  {BAD} no .env file at {env}")
        print("        fix: copy .env.example to .env, then set ODDS_API_KEY")
        return 1
    print(f"  {OK} found {env}")

    key = credentials().get("odds_api_key")
    if not key:
        print(f"  {BAD} ODDS_API_KEY is empty in .env")
        print("        get a free key (email only) at https://the-odds-api.com/")
        return 1
    print(f"  {OK} ODDS_API_KEY set: {key[:4]}...{key[-3:]} ({len(key)} chars)")

    gi = ROOT / ".gitignore"
    if gi.exists() and ".env" in gi.read_text(encoding="utf-8"):
        print(f"  {OK} .env is gitignored")
    else:
        print(f"  {WARN} .env may not be gitignored - check before committing")

    # ------------------------------------------------------------ stage 1
    print("\nStage 1 - API call")
    cfg = params()["oddsapi"]
    client = OddsAPIClient()
    try:
        events = client.get_odds(
            regions=cfg["regions"], markets=cfg["markets"],
            max_age=0 if args.fresh else float(cfg["cache_seconds"]),
        )
    except OddsAPIError as exc:
        print(f"  {BAD} {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"  {BAD} {type(exc).__name__}: {exc}")
        return 1

    if client.last_cost == 0:
        print(f"  {OK} served from cache (0 credits spent)")
        print("        re-run with --fresh to force a live call")
    else:
        print(f"  {OK} live call succeeded, cost {client.last_cost} credits")
    print(f"        {client.quota()}")
    if client.requests_remaining is not None and client.requests_remaining < 50:
        print(f"  {WARN} only {client.requests_remaining} credits left this month")

    # ------------------------------------------------------------ stage 2
    print("\nStage 2 - Premier League coverage")
    if not events:
        print(f"  {BAD} no events returned (international break, or season gap?)")
        return 1
    print(f"  {OK} {len(events)} upcoming fixtures with odds")
    print(f"        earliest {events[0].get('commence_time')}")

    books = Counter(b["key"] for e in events for b in e.get("bookmakers", []))
    print("\n  bookmakers present:")
    for k, n in books.most_common():
        tag = ""
        if k in EXCHANGE_PREFERENCE:
            rank = EXCHANGE_PREFERENCE.index(k)
            tag = "  <-- PREFERRED" if rank == 0 else f"  <-- fallback #{rank}"
        print(f"     {k:<22} {n:>3} fixtures{tag}")

    if "betfair_ex_uk" not in books:
        print(f"\n  {WARN} betfair_ex_uk absent - falling back down the preference list.")
        print("        Exchange prices are what this model is built on; if no exchange")
        print("        appears at all, the numbers carry a bookmaker margin.")
    else:
        pct = 100 * books["betfair_ex_uk"] / len(events)
        print(f"\n  {OK} Betfair Exchange covers {books['betfair_ex_uk']}/{len(events)} "
              f"fixtures ({pct:.0f}%)")

    markets = Counter(m["key"] for e in events for b in e.get("bookmakers", [])
                      for m in b.get("markets", []))
    print(f"  markets returned: {dict(markets)}")

    # ------------------------------------------------------------ stage 3
    print("\nStage 3 - fixture join onto FPL")
    fpl = FPLClient()
    boot = fpl.bootstrap()
    fixtures = build_fixtures(fpl.fixtures(), boot)
    event = args.event or fpl.target_event()
    teams = {t["id"]: t for t in boot["teams"]}
    gw = fixtures[fixtures["event"] == event].copy()
    gw["team_h_name"] = gw["team_h"].map(lambda t: teams[t]["name"])
    gw["team_a_name"] = gw["team_a"].map(lambda t: teams[t]["name"])

    mapping, unmatched = match_fixtures(gw, events)
    print(f"  GW{event}: {len(mapping)}/{len(gw)} fixtures matched")
    if unmatched:
        print(f"  {WARN} UNMATCHED (these silently fall back to the model):")
        for u in unmatched:
            print("     -", u)
        print("        if a club name is the problem, add it to TEAM_ALIASES")
        print("        in fplfh/naming.py")
    else:
        print(f"  {OK} every fixture joined")

    if not mapping:
        return 1

    # ------------------------------------------------------------ stage 4
    print("\nStage 4 - scoreline fits")
    by_id = {str(e["id"]): e for e in events}
    ok = 0
    for fid, ev_id in mapping.items():
        fx = gw[gw["fixture_id"] == fid].iloc[0]
        dist, note = scoreline_from_event(
            by_id[ev_id], fx["team_h_name"], fx["team_a_name"]
        )
        label = f"{fx['team_h_short']} v {fx['team_a_short']}"
        if dist is None:
            print(f"  {BAD} {label:<14} {note}")
            continue
        ok += 1
        print(f"  {OK} {label:<14} xG {dist.lambda_home:.2f}-{dist.lambda_away:.2f}  "
              f"P(H/D/A) {dist.p_win(True):.2f}/{dist.p_draw():.2f}/{dist.p_win(False):.2f}  "
              f"CS {dist.p_clean_sheet(True):.2f}/{dist.p_clean_sheet(False):.2f}  "
              f"[{note}]")
    print(f"\n  {ok}/{len(mapping)} fixtures priced from market data")

    # ------------------------------------------------------------ stage 5
    print("\nStage 5 - do independent exchanges agree?")
    print("  Liquid markets should agree closely. Divergence means at least one")
    print("  book is thin or stale, and that fixture deserves less confidence.")
    cmp_df = compare_sources(gw, client)
    if cmp_df.empty:
        print(f"  {WARN} only one source available - no cross-check possible")
    else:
        spread = (cmp_df.groupby("fixture")["xG_home"].agg(["min", "max", "count"]))
        spread["disagreement"] = spread["max"] - spread["min"]
        multi = spread[spread["count"] > 1].sort_values("disagreement", ascending=False)
        if multi.empty:
            print(f"  {WARN} no fixture had two priced sources")
        else:
            print(f"  {OK} {len(multi)} fixtures priced by 2+ sources")
            print(f"        worst xG disagreement: {multi['disagreement'].iloc[0]:.3f} "
                  f"({multi.index[0]})")
            print(f"        median disagreement:   {multi['disagreement'].median():.3f}")
            print("\n" + multi[["count", "min", "max", "disagreement"]].round(3).to_string())

    print("\n" + "=" * 72)
    print("Setup is working. Next:")
    print("    from fplfh.pipeline import run")
    print("    res = run()                      # picks up the key automatically")
    print("    res.fixture_table()              # 'source' column shows oddsapi:...")
    if client.requests_remaining is not None:
        print(f"\n{client.quota()}")
    else:
        print("\n(served from cache - run with --fresh to see live quota)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
