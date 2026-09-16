"""Fixture outlook - how good is each team's run of games?

Answers "who has the best fixtures over the next N gameweeks", which drives most
medium-term FPL decisions: who to buy, who to hold, when to use a chip.

### What difficulty means here

Not FPL's own 1-5 difficulty rating, which is a coarse integer assigned before
the season and never updated. Difficulty here is read off the same scoreline
model everything else uses, so it is expressed in the units that actually score
points:

    xg_for         goals the team is expected to score   -> attackers
    p_clean_sheet  chance of conceding nothing           -> defenders
    xg_against     goals expected against them           -> keepers, DefCon

### Where the numbers come from

Odds reach about three to four gameweeks ahead (24 days on the live feed, which
spanned two gameweeks across an international break). Beyond that the ratings
model fills in. Every row is tagged with which, because a market-priced fixture
deserves more confidence than a modelled one.

### Discounting

Distant gameweeks are discounted, but **not because they are less predictable**.
Measured on a full season, the correlation between a team's attacking rating and
its actual goals is flat at ~0.28-0.31 whether you look one gameweek ahead or
ten - team strength is broadly stable, so a fixture in week 10 is forecast about
as well as one in week 1.

The discount is about **actionability**. You will make transfers between now and
then, so a distant fixture should influence today's decision less. That makes the
rate a planning preference rather than a statistical constant, which is why it is
a parameter rather than a fitted value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model.fallback import fallback_scorelines, team_ratings
from .model.scoreline import ScorelineDistribution

# A planning discount, not a forecast decay. 1.0 weighs every gameweek in the
# window equally; 0.8 concentrates hard on the next couple.
DEFAULT_DISCOUNT = 0.85


def scorelines_for_range(
    client,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    start_event: int,
    n_events: int = 10,
    use_odds: bool = True,
    verbose: bool = False,
) -> tuple[dict[int, ScorelineDistribution], dict[int, str]]:
    """Scoreline distributions for every fixture in a window of gameweeks.

    Market prices are used where the odds feed reaches, and the ratings model
    fills the rest. Returns the distributions and a per-fixture source tag.
    """
    events = list(range(start_event, start_event + n_events))
    window = fixtures[fixtures["event"].isin(events)]
    if window.empty:
        return {}, {}

    # Ratings are estimated once, from everything played so far.
    ratings = team_ratings(players, fixtures, start_event)
    scorelines = fallback_scorelines(window, ratings)
    sources = {fid: "model" for fid in scorelines}

    if use_odds:
        try:
            from .config import params
            from .odds import OddsAPIClient, scorelines_for_gameweek

            teams = {t["id"]: t for t in client.bootstrap()["teams"]}
            named = window.copy()
            named["team_h_name"] = named["team_h"].map(lambda t: teams[t]["name"])
            named["team_a_name"] = named["team_a"].map(lambda t: teams[t]["name"])

            oc = OddsAPIClient()
            if oc.configured:
                # One request covers the whole horizon, so this costs no more
                # than pricing a single gameweek.
                priced, _ = scorelines_for_gameweek(
                    named, oc, max_age=float(params()["oddsapi"]["cache_seconds"])
                )
                scorelines.update(priced)
                for fid in priced:
                    sources[fid] = "market"
                if verbose:
                    print(f"odds covered {len(priced)}/{len(window)} fixtures "
                          f"in GW{start_event}-{start_event + n_events - 1}")
        except Exception as exc:  # noqa: BLE001 - never let pricing kill the outlook
            if verbose:
                print(f"odds unavailable ({type(exc).__name__}); using ratings only")
    return scorelines, sources


def build_outlook(
    client,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    start_event: int,
    n_events: int = 10,
    use_odds: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """One row per (team, fixture) across the window, with difficulty measures.

    Doubles give a team two rows in a gameweek; blanks give none, which is what
    makes them show up correctly in the aggregates.
    """
    boot = client.bootstrap()
    names = {t["id"]: t["short_name"] for t in boot["teams"]}
    events = list(range(start_event, start_event + n_events))
    window = fixtures[fixtures["event"].isin(events)]

    scorelines, sources = scorelines_for_range(
        client, players, fixtures, start_event, n_events, use_odds, verbose
    )

    rows = []
    for _, fx in window.iterrows():
        d = scorelines.get(int(fx["fixture_id"]))
        if d is None:
            continue
        for is_home in (True, False):
            team = fx["team_h"] if is_home else fx["team_a"]
            opp = fx["team_a"] if is_home else fx["team_h"]
            rows.append({
                "team": names[team],
                "team_id": int(team),
                "event": int(fx["event"]),
                "opponent": names[opp],
                "is_home": is_home,
                "venue": "H" if is_home else "A",
                "xg_for": d.expected_goals(is_home),
                "xg_against": d.expected_conceded(is_home),
                "p_clean_sheet": d.p_clean_sheet(is_home),
                "p_win": d.p_win(is_home),
                "source": sources.get(int(fx["fixture_id"]), "model"),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["horizon"] = out["event"] - start_event + 1
    return out


def fixture_grid(outlook: pd.DataFrame, value: str = "opponent") -> pd.DataFrame:
    """Teams down the side, gameweeks across the top.

    ``value`` picks what fills each cell: ``opponent`` for a readable schedule,
    or any numeric column (``xg_for``, ``p_clean_sheet``, ...) for a heat map.
    Doubles are joined with ``+``; blanks are left empty.
    """
    if outlook.empty:
        return pd.DataFrame()
    if value == "opponent":
        outlook = outlook.copy()
        outlook["_cell"] = outlook["opponent"] + " (" + outlook["venue"] + ")"
        grid = outlook.pivot_table(
            index="team", columns="event", values="_cell",
            aggfunc=lambda s: " + ".join(s),
        )
        return grid.fillna("-")
    grid = outlook.pivot_table(index="team", columns="event", values=value, aggfunc="sum")
    return grid


def rank_fixtures(
    outlook: pd.DataFrame,
    window: int = 5,
    discount: float = DEFAULT_DISCOUNT,
    metric: str = "attack",
) -> pd.DataFrame:
    """Rank teams by fixture quality over the next ``window`` gameweeks.

    ``metric`` chooses whose fixtures you care about:

    ``attack``   expected goals for - for forwards and attacking midfielders
    ``defence``  clean sheet probability - for defenders and keepers
    ``overall``  both, standardised and averaged

    ``discount`` weights gameweek k by ``discount ** (k - 1)``. At 1.0 every
    gameweek in the window counts equally; lower values concentrate on the
    near term. See the module docstring for why this is a planning choice and
    not a measured decay.

    Scores are normalised by the total weight, so a team with a blank gameweek
    is correctly penalised rather than flattered by having one fewer fixture.
    """
    if outlook.empty:
        return pd.DataFrame()

    sub = outlook[outlook["horizon"] <= window].copy()
    sub["weight"] = discount ** (sub["horizon"] - 1)

    # Every team is charged the full weight of the window, whether or not it has
    # a fixture. Otherwise a blank gameweek would raise a team's average.
    full_weight = sum(discount ** (k - 1) for k in range(1, window + 1))

    g = sub.groupby("team")
    out = pd.DataFrame({
        "fixtures": g.size(),
        "blanks": window - g["event"].nunique(),
        "doubles": g["event"].apply(lambda s: int((s.value_counts() > 1).sum())),
        "xg_for": g.apply(lambda d: (d.xg_for * d.weight).sum() / full_weight,
                          include_groups=False),
        "xg_against": g.apply(lambda d: (d.xg_against * d.weight).sum() / full_weight,
                              include_groups=False),
        "clean_sheet": g.apply(lambda d: (d.p_clean_sheet * d.weight).sum() / full_weight,
                               include_groups=False),
        "home_games": g["is_home"].sum(),
        "market_priced": g["source"].apply(lambda s: int((s == "market").sum())),
    })

    if metric == "attack":
        out["score"] = out["xg_for"]
    elif metric == "defence":
        out["score"] = out["clean_sheet"]
    elif metric == "overall":
        z = lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else s * 0
        out["score"] = (z(out["xg_for"]) + z(out["clean_sheet"])) / 2
    else:
        raise ValueError(f"metric must be attack, defence or overall - got {metric!r}")

    out["opponents"] = sub.sort_values("event").groupby("team").apply(
        lambda d: ", ".join(f"{o}({v})" for o, v in zip(d.opponent, d.venue)),
        include_groups=False,
    )
    return out.sort_values("score", ascending=False)


def compare_windows(outlook: pd.DataFrame, windows=(3, 5, 8),
                    discount: float = DEFAULT_DISCOUNT,
                    metric: str = "attack") -> pd.DataFrame:
    """Rank over several windows at once.

    Teams whose rank swings between windows are the interesting ones - a good
    short run followed by a hard patch, or vice versa.
    """
    frames = {}
    for w in windows:
        r = rank_fixtures(outlook, window=w, discount=discount, metric=metric)
        frames[f"next_{w}"] = r["score"].rank(ascending=False).astype(int)
    out = pd.DataFrame(frames)
    out["swing"] = out.max(axis=1) - out.min(axis=1)
    return out.sort_values(out.columns[0])
