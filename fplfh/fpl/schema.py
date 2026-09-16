"""Normalise raw FPL JSON into tidy DataFrames.

The central table is ``player_fixtures``: one row per (player, fixture) pair.
Building it this way means double and blank gameweeks are handled by
construction - a player with two fixtures gets two rows and their expected
points sum; a player whose team is blank gets no rows and scores zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ELEMENT_TYPE_TO_POS


def _safe_div(a, b, fill=0.0):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), fill)


def build_players(bootstrap: dict) -> pd.DataFrame:
    """One row per player, with per-90 rates and availability."""
    teams = {t["id"]: t for t in bootstrap["teams"]}
    rows = []
    for e in bootstrap["elements"]:
        mins = float(e["minutes"])
        nineties = mins / 90.0
        rows.append(
            {
                "player_id": e["id"],
                "web_name": e["web_name"],
                "full_name": f"{e['first_name']} {e['second_name']}".strip(),
                "team_id": e["team"],
                "team": teams[e["team"]]["short_name"],
                "team_name": teams[e["team"]]["name"],
                "position": ELEMENT_TYPE_TO_POS[e["element_type"]],
                "price": e["now_cost"] / 10.0,
                "cost_tenths": e["now_cost"],
                "selected_by_percent": float(e["selected_by_percent"]),
                # --- availability -------------------------------------------
                "status": e["status"],
                "chance_of_playing": e.get("chance_of_playing_next_round"),
                "news": e.get("news", "") or "",
                # --- volume -------------------------------------------------
                "minutes": mins,
                "starts": int(e["starts"]),
                "nineties": nineties,
                # --- per-90 rates (Opta-derived where available) -------------
                "xg90": float(e.get("expected_goals_per_90") or 0.0),
                "xa90": float(e.get("expected_assists_per_90") or 0.0),
                "xgc90": float(e.get("expected_goals_conceded_per_90") or 0.0),
                "dc90": float(e.get("defensive_contribution_per_90") or 0.0),
                "saves90": float(e.get("saves_per_90") or 0.0),
                "yc90": _safe_div(e["yellow_cards"], nineties).item(),
                # --- raw totals, for shrinkage -------------------------------
                "xg_total": float(e.get("expected_goals") or 0.0),
                "xa_total": float(e.get("expected_assists") or 0.0),
                "dc_total": int(e.get("defensive_contribution") or 0),
                "saves_total": int(e.get("saves") or 0),
                "goals": int(e["goals_scored"]),
                "assists": int(e["assists"]),
                "yellow_cards": int(e["yellow_cards"]),
                "bonus": int(e["bonus"]),
                "total_points": int(e["total_points"]),
                # --- set pieces ----------------------------------------------
                "pens_order": e.get("penalties_order"),
                "corners_order": e.get("corners_and_indirect_freekicks_order"),
                "fk_order": e.get("direct_freekicks_order"),
                "is_pen_taker": (e.get("penalties_order") == 1),
            }
        )
    df = pd.DataFrame(rows)
    # Drop players removed from the game entirely.
    df = df[~df["status"].isin(["n"]) | (df["minutes"] > 0)].reset_index(drop=True)
    return df


def build_fixtures(fixtures: list[dict], bootstrap: dict) -> pd.DataFrame:
    """One row per fixture."""
    teams = {t["id"]: t for t in bootstrap["teams"]}
    rows = []
    for f in fixtures:
        if f.get("event") is None:
            continue  # not yet scheduled to a gameweek
        rows.append(
            {
                "fixture_id": f["id"],
                "event": int(f["event"]),
                "kickoff": pd.to_datetime(f["kickoff_time"], utc=True, errors="coerce"),
                "finished": bool(f["finished"]),
                "team_h": f["team_h"],
                "team_a": f["team_a"],
                "team_h_short": teams[f["team_h"]]["short_name"],
                "team_a_short": teams[f["team_a"]]["short_name"],
                "team_h_difficulty": f.get("team_h_difficulty"),
                "team_a_difficulty": f.get("team_a_difficulty"),
            }
        )
    return pd.DataFrame(rows).sort_values(["event", "kickoff"]).reset_index(drop=True)


def build_player_fixtures(
    players: pd.DataFrame, fixtures: pd.DataFrame, event: int
) -> pd.DataFrame:
    """Cross players with their team's fixtures in ``event``.

    Doubles produce two rows per player; blanks produce none.
    """
    gw = fixtures[fixtures["event"] == event]
    if gw.empty:
        raise ValueError(f"No fixtures found for gameweek {event}")

    long = pd.concat(
        [
            gw.assign(team_id=gw["team_h"], opponent_id=gw["team_a"], is_home=True,
                      opponent=gw["team_a_short"]),
            gw.assign(team_id=gw["team_a"], opponent_id=gw["team_h"], is_home=False,
                      opponent=gw["team_h_short"]),
        ],
        ignore_index=True,
    )[["fixture_id", "event", "kickoff", "team_id", "opponent_id", "opponent", "is_home"]]

    pf = players.merge(long, on="team_id", how="inner")
    pf["n_fixtures_in_gw"] = pf.groupby("player_id")["fixture_id"].transform("size")
    return pf.sort_values(["player_id", "kickoff"]).reset_index(drop=True)


def gameweek_summary(fixtures: pd.DataFrame, event: int) -> str:
    gw = fixtures[fixtures["event"] == event]
    teams_playing = set(gw["team_h"]) | set(gw["team_a"])
    counts = pd.concat([gw["team_h"], gw["team_a"]]).value_counts()
    doubles = (counts > 1).sum()
    return (
        f"GW{event}: {len(gw)} fixtures, {len(teams_playing)}/20 teams playing"
        f"{f', {doubles} team(s) with a double' if doubles else ''}"
    )
