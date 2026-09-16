"""Scoreline model for fixtures with no usable exchange data.

Used when no market exists for a fixture, when liquidity is too thin to
trust, or when running entirely offline. It is deliberately a *separate*
estimate rather than a blend, so that output provenance stays honest: a
fixture priced this way is tagged ``model:xg_ratings`` and never claims to be
exchange-derived.

Team strength is built from Opta expected goals rather than actual goals.
Early in a season that choice matters a great deal - after four matches, goals
scored is mostly noise, while xG has several times the sample behind it (every
shot, not every goal) and is far more predictive of what comes next.

Ratings are multiplicative and shrunk toward the league average in proportion
to how little has been played, so in gameweek 2 every team looks close to
average and separation emerges only as evidence accumulates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import params
from .scoreline import ScorelineDistribution, from_lambdas


def team_ratings(
    players: pd.DataFrame, fixtures: pd.DataFrame, upto_event: int, shrink_games: float = 6.0
) -> pd.DataFrame:
    """Multiplicative attack and defence ratings per team, centred on 1.0."""
    # Games played is derived from player minutes, not from the fixture
    # `finished` flag. The flag lags: it stays False until FPL has processed a
    # gameweek, while player stats update as matches are played. Trusting it
    # divides four gameweeks of xG by three gameweeks of fixtures and inflates
    # every attack rating by a third.
    rows = []
    for team_id, grp in players.groupby("team_id"):
        n = float(grp["minutes"].max() / 90.0) if len(grp) else 0.0
        n = max(round(n), 0)
        # Attack: player xG is additive - each shot belongs to exactly one player.
        team_xg = float(grp["xg_total"].sum())
        # Defence: xG *conceded* is a team property seen by everyone on the
        # pitch, so summing players would count the same shots eleven times.
        # The keeper is on the pitch for essentially the whole match, so their
        # per-90 rate is the cleanest read on the team's rate.
        gks = grp[(grp["position"] == "GKP") & (grp["minutes"] > 0)]
        if len(gks):
            top_gk = gks.loc[gks["minutes"].idxmax()]
            team_xgc_pg = float(top_gk["xgc90"])
        else:
            outfield = grp[grp["minutes"] > 200]
            team_xgc_pg = float(outfield["xgc90"].median()) if len(outfield) else np.nan
        rows.append(
            {
                "team_id": team_id,
                "team": grp["team"].iloc[0],
                "games": n,
                "xg_per_game": team_xg / n if n > 0 else np.nan,
                "xgc_per_game": team_xgc_pg,
            }
        )

    df = pd.DataFrame(rows)
    league_xg = float(np.nanmean(df["xg_per_game"]))
    league_xgc = float(np.nanmean(df["xgc_per_game"]))
    if not np.isfinite(league_xg) or league_xg <= 0:
        league_xg = float(params()["scoreline"]["fallback_league_avg_goals"])
    if not np.isfinite(league_xgc) or league_xgc <= 0:
        league_xgc = league_xg

    raw_att = df["xg_per_game"] / league_xg
    raw_def = df["xgc_per_game"] / league_xgc

    # Shrink toward 1.0: with few games played, nobody is far from average.
    w = (df["games"] / (df["games"] + shrink_games)).fillna(0.0)
    df["attack"] = (w * raw_att.fillna(1.0) + (1 - w) * 1.0).clip(0.4, 2.2)
    df["defence"] = (w * raw_def.fillna(1.0) + (1 - w) * 1.0).clip(0.4, 2.2)
    df["league_xg"] = league_xg
    return df


def fallback_scorelines(
    fixtures_gw: pd.DataFrame, ratings: pd.DataFrame
) -> dict[int, ScorelineDistribution]:
    """Scoreline distributions for a gameweek, from ratings alone."""
    cfg = params()["scoreline"]
    home_adv = float(np.exp(cfg["fallback_home_advantage"]))
    base = float(ratings["league_xg"].iloc[0]) if len(ratings) else float(
        cfg["fallback_league_avg_goals"]
    )
    r = ratings.set_index("team_id")

    out: dict[int, ScorelineDistribution] = {}
    for _, fx in fixtures_gw.iterrows():
        h, a = int(fx["team_h"]), int(fx["team_a"])
        if h not in r.index or a not in r.index:
            continue
        lh = base * float(r.at[h, "attack"]) * float(r.at[a, "defence"]) * np.sqrt(home_adv)
        la = base * float(r.at[a, "attack"]) * float(r.at[h, "defence"]) / np.sqrt(home_adv)
        out[int(fx["fixture_id"])] = from_lambdas(
            float(np.clip(lh, 0.2, 4.5)),
            float(np.clip(la, 0.2, 4.5)),
            source="model:xg_ratings",
        )
    return out
