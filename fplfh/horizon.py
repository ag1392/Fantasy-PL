"""Expected points across a multi-gameweek window, and transfer suggestions.

Section 7's fixture outlook ranks *teams*. This ranks *players*, which is what a
transfer decision actually needs: fixtures matter, but so do minutes, role and
price.

### What it does

Runs the single-gameweek model once per gameweek in the window, then sums each
player's expected points with a discount on later weeks.

### The assumptions, stated plainly

**Minutes are frozen at today's estimate, except where FPL states a return
date.** Roughly 13% of flagged players carry one ("Expected back 11 Oct"); those
switch back on at the right gameweek. The rest hold today's availability for the
whole window, which is the honest default - we do not know when they return, and
a made-up recovery curve would look like information while being a guess.

A returning player's start rate comes from the (position, price) prior rather
than their own record, because that record is a string of zeros *caused by* the
injury and says nothing about whether they would be picked when fit. Those rows
are tagged ``minutes_basis`` so they can be flagged rather than trusted blindly.

**Player rates are frozen.** Defensible: measured over a full season, a team's
attacking rating predicts its goals about as well ten gameweeks out as one, so
there is no evidence that freezing rates degrades with horizon.

**Market prices only reach two to four gameweeks.** Beyond that the ratings model
fills in, so the far end of a long window is lower confidence. The
``market_share`` column reports how much of each player's total came from priced
fixtures.

**Prices are frozen.** Over a long window they move, which matters for what you
can afford later. Ignored here.

### The discount

Same meaning as in the fixture outlook: not forecast decay, but *actionability*.
You will make transfers before gameweek eight arrives, so a player's value then
should sway today's decision less than their value next week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fpl import build_player_fixtures
from .model import compute_expected_points
from .model.minutes import minutes_for_event
from .outlook import DEFAULT_DISCOUNT, scorelines_for_range


def player_horizon(
    client,
    minutes: pd.DataFrame,
    fixtures: pd.DataFrame,
    start_event: int,
    n_events: int = 6,
    discount: float = DEFAULT_DISCOUNT,
    use_odds: bool = True,
    return_gw: pd.Series | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discounted expected points per player over ``n_events`` gameweeks.

    Returns ``(summary, per_gameweek)``. The summary has one row per player; the
    per-gameweek frame keeps every (player, fixture) row so you can see where a
    total came from.
    """
    events = list(range(start_event, start_event + n_events))
    scorelines, sources = scorelines_for_range(
        client, minutes, fixtures, start_event, n_events, use_odds, verbose
    )
    if not scorelines:
        return pd.DataFrame(), pd.DataFrame()

    frames = []
    for e in events:
        gw_fx = fixtures[fixtures["event"] == e]
        if gw_fx.empty:
            continue
        m = minutes_for_event(minutes, e, return_gw)
        pf = compute_expected_points(build_player_fixtures(m, fixtures, e), scorelines)
        pf["event"] = e
        pf["horizon"] = e - start_event + 1
        pf["weight"] = discount ** (pf["horizon"] - 1)
        pf["priced"] = pf["fixture_id"].map(
            {k: (v == "market") for k, v in sources.items()}
        ).fillna(False)
        frames.append(pf)

    per_gw = pd.concat(frames, ignore_index=True)
    per_gw["xp_weighted"] = per_gw["xp"] * per_gw["weight"]

    keys = ["player_id", "web_name", "team", "team_id", "position", "price", "cost_tenths"]
    g = per_gw.groupby(keys, dropna=False)
    summary = pd.DataFrame({
        "xp_total": g["xp_weighted"].sum(),
        "xp_raw": g["xp"].sum(),
        "xp_next": g.apply(lambda d: d.loc[d.horizon == 1, "xp"].sum(), include_groups=False),
        "fixtures": g.size(),
        "doubles": g["event"].apply(lambda s: int((s.value_counts() > 1).sum())),
        "market_share": g.apply(
            lambda d: float((d.xp * d.priced).sum() / d.xp.sum()) if d.xp.sum() else 0.0,
            include_groups=False,
        ),
        "xmins_mean": g["xmins"].mean(),
    }).reset_index()

    # Blanks: gameweeks in the window where this player's team does not play.
    played = per_gw.groupby("player_id")["event"].nunique()
    summary["blanks"] = n_events - summary["player_id"].map(played).fillna(0).astype(int)

    # Flag anyone whose minutes rest on a prior rather than an observed record.
    if "minutes_basis" in per_gw.columns:
        basis = per_gw.groupby("player_id")["minutes_basis"].apply(
            lambda s: "prior" if (s != "observed").any() else "observed"
        )
        summary["minutes_basis"] = summary["player_id"].map(basis).fillna("observed")

    summary["xp_per_million"] = summary["xp_total"] / summary["price"]
    return summary.sort_values("xp_total", ascending=False).reset_index(drop=True), per_gw


def gameweek_matrix(per_gw: pd.DataFrame, players: list[str] | None = None,
                    value: str = "xp") -> pd.DataFrame:
    """Players down the side, gameweeks across - to see the shape of a run."""
    d = per_gw if players is None else per_gw[per_gw["web_name"].isin(players)]
    return d.pivot_table(index="web_name", columns="event", values=value, aggfunc="sum")


def suggest_transfers(
    horizon: pd.DataFrame,
    squad_ids: list[int],
    bank: float = 0.0,
    max_per_team: int = 3,
    top_n: int = 3,
) -> pd.DataFrame:
    """For each player you own, the best affordable replacements.

    A deliberately simple, greedy view: it evaluates each swap on its own and
    does not consider combinations, hit costs, or the fact that selling two
    players frees more budget than selling one.

    Constraints respected: position must match, the buy must be affordable from
    that player's sale price plus the bank, and the incoming player must not
    breach the three-per-club limit.

    Sale price is approximated as the current price. FPL returns only half of
    any profit made since purchase, so real funds can be slightly lower.
    """
    owned = horizon[horizon["player_id"].isin(squad_ids)]
    if owned.empty:
        return pd.DataFrame()

    club_counts = owned["team"].value_counts().to_dict()
    rows = []
    for _, out_p in owned.iterrows():
        budget = out_p["price"] + bank
        # Selling frees a slot at the outgoing player's club.
        after_sale = dict(club_counts)
        after_sale[out_p["team"]] = after_sale.get(out_p["team"], 0) - 1

        # Clubs with room for another player once the outgoing one is sold.
        # Built as a set and applied with .isin() rather than .map(), because an
        # empty .map() returns an object-dtype Series, which pandas then treats
        # as a *column selector* instead of a boolean mask - silently dropping
        # every column whenever a position has no candidates.
        allowed_clubs = {
            t for t in horizon["team"].unique()
            if after_sale.get(t, 0) < max_per_team
        }
        legal = horizon[
            (horizon["position"] == out_p["position"])
            & (~horizon["player_id"].isin(squad_ids))
            & (horizon["price"] <= budget + 1e-9)
            & (horizon["xp_total"] > out_p["xp_total"])
            & (horizon["team"].isin(allowed_clubs))
        ]
        if legal.empty:
            continue
        for _, in_p in legal.nlargest(top_n, "xp_total").iterrows():
            rows.append({
                "out": out_p["web_name"], "out_team": out_p["team"],
                "out_price": out_p["price"], "out_xp": round(out_p["xp_total"], 2),
                "in": in_p["web_name"], "in_team": in_p["team"],
                "in_price": in_p["price"], "in_xp": round(in_p["xp_total"], 2),
                "gain": round(in_p["xp_total"] - out_p["xp_total"], 2),
                "cost": round(in_p["price"] - out_p["price"], 1),
                "in_minutes_basis": in_p.get("minutes_basis", "observed"),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)


def transfer_summary(suggestions: pd.DataFrame, bank: float = 0.0,
                     squad_teams: list[str] | None = None,
                     free_transfers: int = 1, hit_cost: float = 4.0,
                     max_per_team: int = 3) -> pd.DataFrame:
    """An ordered, *self-consistent* transfer plan.

    Each outgoing player appears once and - importantly - each incoming player
    appears once too. Taking the best upgrade for every player independently
    produces a table that recommends buying the same bargain four times over,
    which reads like a plan but is not one.

    Assignment is greedy on gain: the biggest single upgrade is taken first,
    both players are struck off, and the next best swap among what remains is
    taken, and so on.

    The plan is also checked for *cumulative* feasibility. Each individual swap
    is affordable on its own, but doing them in sequence moves the bank and the
    club counts, so a list that looks fine swap-by-swap can be impossible to
    execute. Both are tracked as the plan is built and infeasible swaps are
    skipped.

    ``net`` is the gain after the hit. Two caveats worth keeping in view: the
    gain is a *whole-window* total while the hit is paid once, and a net under
    about a point is well inside the model's own error.
    """
    if suggestions.empty:
        return pd.DataFrame()

    running_bank = float(bank)
    clubs: dict[str, int] = {}
    for t in (squad_teams or []):
        clubs[t] = clubs.get(t, 0) + 1

    plan, used_out, used_in = [], set(), set()
    for _, row in suggestions.sort_values("gain", ascending=False).iterrows():
        if row["out"] in used_out or row["in"] in used_in:
            continue
        cost = float(row["in_price"]) - float(row["out_price"])
        if cost > running_bank + 1e-9:
            continue                       # cannot afford it once earlier swaps are done
        if squad_teams is not None:
            after = clubs.get(row["in_team"], 0) + 1 - (
                1 if row["in_team"] == row["out_team"] else 0
            )
            if after > max_per_team:
                continue                   # would breach the three-per-club limit
            clubs[row["out_team"]] = clubs.get(row["out_team"], 1) - 1
            clubs[row["in_team"]] = clubs.get(row["in_team"], 0) + 1
        running_bank -= cost
        row = row.copy()
        row["bank_after"] = round(running_bank, 1)
        plan.append(row)
        used_out.add(row["out"])
        used_in.add(row["in"])

    if not plan:
        return pd.DataFrame()
    best = pd.DataFrame(plan).reset_index(drop=True)
    best["transfer_no"] = best.index + 1
    best["hit"] = np.where(best["transfer_no"] <= free_transfers, 0.0, hit_cost)
    best["net"] = best["gain"] - best["hit"]
    best["cumulative_net"] = best["net"].cumsum()
    cols = ["transfer_no", "out", "out_team", "out_price", "out_xp",
            "in", "in_team", "in_price", "in_xp", "gain", "hit", "net",
            "cumulative_net", "cost", "in_minutes_basis"]
    if "bank_after" in best.columns:
        cols.insert(-1, "bank_after")
    return best[cols]
