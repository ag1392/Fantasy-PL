"""End-to-end run: FPL data + exchange prices -> expected points -> squad.

Wraps the whole flow so a notebook cell can be one line, while every
intermediate table stays reachable on the returned object for inspection.

Market pricing is optional and degrades gracefully. With an API key, fixtures
with usable odds are priced from the exchange and the rest fall back to the xG
ratings model; without a key everything falls back. Either way the per-fixture
``xp_source`` column records which path was taken, so the output never
overstates how much of itself is market-derived.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import ensure_dirs, params
from .fpl import FPLClient, build_fixtures, build_player_fixtures, build_players
from .model import (
    aggregate_to_players,
    build_minutes_history,
    compute_expected_points,
    estimate_minutes,
    fallback_scorelines,
    provenance_summary,
    team_ratings,
)
from .scoring import ScoringRules


@dataclass
class RunResult:
    """Everything a run produces, so nothing is hidden inside the pipeline."""

    event: int
    players: pd.DataFrame            # one row per player for the gameweek
    player_fixtures: pd.DataFrame    # one row per (player, fixture)
    scorelines: dict
    fixtures: pd.DataFrame
    rules: ScoringRules
    minutes: pd.DataFrame
    ratings: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    @property
    def provenance(self) -> pd.DataFrame:
        return provenance_summary(self.player_fixtures)

    def exchange_share(self) -> float:
        """Fraction of total expected points resting on exchange-priced fixtures."""
        pf = self.player_fixtures
        total = pf["xp"].sum()
        if total <= 0:
            return 0.0
        market = pf["xp_source"].str.startswith("oddsapi")
        return float(pf.loc[market, "xp"].sum() / total)

    def fixture_table(self) -> pd.DataFrame:
        """Per-fixture view of the scoreline model."""
        rows = []
        for _, fx in self.fixtures[self.fixtures["event"] == self.event].iterrows():
            d = self.scorelines.get(int(fx["fixture_id"]))
            if d is None:
                continue
            rows.append(
                {
                    "fixture": f"{fx['team_h_short']} v {fx['team_a_short']}",
                    "kickoff": fx["kickoff"],
                    "xG_home": round(d.lambda_home, 2),
                    "xG_away": round(d.lambda_away, 2),
                    "P(home)": round(d.p_win(True), 3),
                    "P(draw)": round(d.p_draw(), 3),
                    "P(away)": round(d.p_win(False), 3),
                    "CS_home": round(d.p_clean_sheet(True), 3),
                    "CS_away": round(d.p_clean_sheet(False), 3),
                    "source": d.source,
                }
            )
        return pd.DataFrame(rows)


def run(
    event: int | None = None,
    odds_source: str = "auto",
    max_age: float | None = None,
    verbose: bool = True,
) -> RunResult:
    """Run the full pipeline for one gameweek.

    ``odds_source`` selects where market prices come from:

    ``"auto"``     market odds if an API key is configured, else the
                   statistical fallback. Never fails.
    ``"oddsapi"``  market odds only.
    ``"none"``     force the statistical model, for comparison.

    """
    ensure_dirs()
    warnings: list[str] = []

    fpl = FPLClient()
    boot = fpl.bootstrap(max_age=max_age)
    event = event or fpl.target_event()

    rules = ScoringRules.from_bootstrap(boot)
    warnings.extend(rules.warnings)

    players = build_players(boot)
    fixtures = build_fixtures(fpl.fixtures(max_age=max_age), boot)

    minutes = estimate_minutes(players, build_minutes_history(fpl), event)
    for w in minutes.attrs.get("override_warnings", []):
        warnings.append(f"minutes override not matched: {w}")

    # --- scorelines: market prices first, model for the remainder -----------
    ratings = team_ratings(minutes, fixtures, event)
    gw_fixtures = fixtures[fixtures["event"] == event]
    scorelines = fallback_scorelines(gw_fixtures, ratings)
    anytime: dict = {}

    teams = {t["id"]: t for t in boot["teams"]}
    gw_named = gw_fixtures.copy()
    gw_named["team_h_name"] = gw_named["team_h"].map(lambda t: teams[t]["name"])
    gw_named["team_a_name"] = gw_named["team_a"].map(lambda t: teams[t]["name"])

    if odds_source in ("auto", "oddsapi"):
        try:
            from .odds import OddsAPIClient, scorelines_for_gameweek

            client = OddsAPIClient()
            if not client.configured:
                raise RuntimeError("ODDS_API_KEY not set")
            oa, oa_notes = scorelines_for_gameweek(
                gw_named, client, max_age=float(params()["oddsapi"]["cache_seconds"])
            )
            scorelines.update(oa)
            warnings.extend(oa_notes)
            if verbose:
                print(f"Odds API: priced {len(oa)}/{len(gw_fixtures)} fixtures  "
                      f"[{client.quota()}]")
        except Exception as exc:  # noqa: BLE001 - pricing must never kill a run
            warnings.append(f"Odds API unavailable ({exc})")
            if verbose:
                print(f"Odds API unavailable ({type(exc).__name__}: {exc})")


    pf = compute_expected_points(
        build_player_fixtures(minutes, fixtures, event), scorelines, anytime or None
    )
    agg = aggregate_to_players(pf)
    # Carry minutes fields through for the optimiser and for review.
    agg = agg.merge(
        minutes[["player_id", "xg90", "xa90", "dc90", "chance_of_playing"]],
        on="player_id", how="left",
    )

    if verbose:
        print(f"GW{event}: {len(agg)} players, {len(gw_fixtures)} fixtures")
        for w in warnings:
            print("  warning:", w)

    return RunResult(
        event=event,
        players=agg,
        player_fixtures=pf,
        scorelines=scorelines,
        fixtures=fixtures,
        rules=rules,
        minutes=minutes,
        ratings=ratings,
        warnings=warnings,
    )

