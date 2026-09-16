"""Backtesting: what would the tool have picked, and what did it actually score?

The whole value of a backtest rests on one thing - **not letting the model see
the future**. That is harder than it looks here, because the live FPL API serves
*season-to-date* totals. Running today's pipeline against gameweek 3 would let
the model use gameweek 3 and 4 data to predict gameweek 3, and it would look
wonderful for entirely fake reasons.

So the player state is rebuilt from scratch out of per-gameweek history
(``element-summary``), summing only gameweeks strictly before the target, and
taking each player's price *as it was at that gameweek* rather than today's.

### What is deliberately withheld

Availability flags (``status``, ``chance_of_playing``) are **current state** -
the API has no history for them. Using today's values would leak: a player still
injured now, who was also injured then, would be correctly avoided for the wrong
reason.

They are therefore withheld entirely, and every player is treated as fully
available. This makes the backtest a **lower bound** on the live tool, which does
get to see team news. Expect the backtested squad to contain someone who was
visibly injured at the time; that is the cost of an honest test.

### What still leaks, mildly

Set-piece orders (penalties, corners) are current-state and are kept, because
they are fairly stable across a season and only feed the penalty-miss term,
which is worth a few hundredths of a point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ELEMENT_TYPE_TO_POS

# Columns the rest of the pipeline expects from `build_players`.
_SET_PIECE = ("pens_order", "corners_order", "fk_order")


def fetch_histories(client, player_ids=None, verbose=True) -> pd.DataFrame:
    """Per-gameweek history for every player, as one tidy frame.

    Hits ``element-summary`` once per player. Slow on a cold cache (~650 calls),
    instant afterwards.
    """
    boot = client.bootstrap()
    ids = list(player_ids) if player_ids is not None else [e["id"] for e in boot["elements"]]
    rows = []
    for n, pid in enumerate(ids, 1):
        if verbose and n % 100 == 0:
            print(f"  {n}/{len(ids)} players...")
        try:
            hist = client.element_summary(pid)["history"]
        except Exception:  # noqa: BLE001 - a missing player must not kill the run
            continue
        for h in hist:
            rows.append({"player_id": pid, **h})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.rename(columns={"round": "event"})
    return df


def reconstruct_players(
    histories: pd.DataFrame, bootstrap: dict, target_event: int
) -> pd.DataFrame:
    """Player state as it stood *before* ``target_event``.

    Uses only gameweeks strictly earlier than the target, and prices as at the
    target gameweek. Column-compatible with ``fpl.schema.build_players`` so the
    rest of the pipeline runs unchanged.
    """
    meta = {e["id"]: e for e in bootstrap["elements"]}
    teams = {t["id"]: t for t in bootstrap["teams"]}

    past = histories[histories["event"] < target_event]
    if past.empty:
        raise ValueError(
            f"no history before gameweek {target_event} - cannot backtest the "
            "opening gameweek, there is nothing to build a prior from"
        )

    num = [
        "minutes", "starts", "goals_scored", "assists", "yellow_cards", "red_cards",
        "saves", "bonus", "total_points", "clean_sheets", "goals_conceded",
        "defensive_contribution", "tackles", "clearances_blocks_interceptions",
        "recoveries",
    ]
    fl = ["expected_goals", "expected_assists", "expected_goals_conceded"]
    for c in num + fl:
        if c in past.columns:
            past = past.assign(**{c: pd.to_numeric(past[c], errors="coerce").fillna(0.0)})

    agg = past.groupby("player_id").agg({c: "sum" for c in num if c in past.columns})
    for c in fl:
        if c in past.columns:
            agg[c] = past.groupby("player_id")[c].sum()

    # Price as at the target gameweek; fall back to the most recent price seen.
    at_target = histories[histories["event"] == target_event].set_index("player_id")["value"]
    last_known = histories.sort_values("event").groupby("player_id")["value"].last()
    price_tenths = at_target.reindex(agg.index).fillna(last_known.reindex(agg.index))

    rows = []
    for pid, r in agg.iterrows():
        m = meta.get(pid)
        if m is None:
            continue
        cost = price_tenths.get(pid)
        if cost is None or not np.isfinite(cost):
            cost = m["now_cost"]
        mins = float(r.get("minutes", 0.0))
        n90 = mins / 90.0
        div = n90 if n90 > 0 else np.nan
        rows.append({
            "player_id": pid,
            "web_name": m["web_name"],
            "full_name": f"{m['first_name']} {m['second_name']}".strip(),
            "team_id": m["team"],
            "team": teams[m["team"]]["short_name"],
            "team_name": teams[m["team"]]["name"],
            "position": ELEMENT_TYPE_TO_POS[m["element_type"]],
            "price": float(cost) / 10.0,
            "cost_tenths": int(cost),
            "selected_by_percent": 0.0,          # current-state; withheld
            # --- availability withheld to avoid look-ahead -------------------
            "status": "a",
            "chance_of_playing": None,
            "news": "",
            # --- volume ------------------------------------------------------
            "minutes": mins,
            "starts": int(r.get("starts", 0)),
            "nineties": n90,
            # --- per-90 rates, from prior gameweeks only ---------------------
            "xg90": float(r.get("expected_goals", 0.0)) / div if div else 0.0,
            "xa90": float(r.get("expected_assists", 0.0)) / div if div else 0.0,
            "xgc90": float(r.get("expected_goals_conceded", 0.0)) / div if div else 0.0,
            "dc90": float(r.get("defensive_contribution", 0.0)) / div if div else 0.0,
            "saves90": float(r.get("saves", 0.0)) / div if div else 0.0,
            "yc90": float(r.get("yellow_cards", 0.0)) / div if div else 0.0,
            # --- totals ------------------------------------------------------
            "xg_total": float(r.get("expected_goals", 0.0)),
            "xa_total": float(r.get("expected_assists", 0.0)),
            "dc_total": int(r.get("defensive_contribution", 0)),
            "saves_total": int(r.get("saves", 0)),
            "goals": int(r.get("goals_scored", 0)),
            "assists": int(r.get("assists", 0)),
            "yellow_cards": int(r.get("yellow_cards", 0)),
            "bonus": int(r.get("bonus", 0)),
            "total_points": int(r.get("total_points", 0)),
            # --- set pieces (mild, documented leak) ---------------------------
            "pens_order": m.get("penalties_order"),
            "corners_order": m.get("corners_and_indirect_freekicks_order"),
            "fk_order": m.get("direct_freekicks_order"),
            "is_pen_taker": (m.get("penalties_order") == 1),
        })
    out = pd.DataFrame(rows)
    for c in (*_SET_PIECE,):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.fillna({"xg90": 0.0, "xa90": 0.0, "dc90": 0.0, "saves90": 0.0, "yc90": 0.0})


def actual_results(client, event: int) -> pd.DataFrame:
    """What actually happened: points and minutes per player for ``event``."""
    live = client.live(event)
    rows = []
    for el in live["elements"]:
        s = el["stats"]
        rows.append({
            "player_id": el["id"],
            "actual_points": int(s["total_points"]),
            "actual_minutes": int(s["minutes"]),
            "played": bool(s["minutes"] > 0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Scoring a squad the way FPL actually scores it
# --------------------------------------------------------------------------
_MIN_PLAY = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
_MAX_PLAY = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}


def _formation_valid(positions: list[str]) -> bool:
    if len(positions) != 11:
        return False
    counts = {p: positions.count(p) for p in ("GKP", "DEF", "MID", "FWD")}
    return all(_MIN_PLAY[p] <= counts[p] <= _MAX_PLAY[p] for p in counts)


def apply_autosubs(squad_df: pd.DataFrame) -> pd.DataFrame:
    """Apply FPL's automatic substitutions.

    A starter who played no minutes is replaced by the first eligible bench
    player who did, provided the resulting formation stays legal. Goalkeepers
    only ever replace goalkeepers.

    Bench order is taken as descending expected points, which is how a manager
    would sensibly set it.
    """
    df = squad_df.copy()
    df["subbed_in"] = False
    df["subbed_out"] = False
    df["counts"] = df["is_starter"]

    bench = df[~df["is_starter"]].sort_values("xp", ascending=False)
    outfield_bench = [i for i in bench.index if df.at[i, "position"] != "GKP"]
    gk_bench = [i for i in bench.index if df.at[i, "position"] == "GKP"]

    # Goalkeeper first - a separate, simpler rule.
    gk_start = df[(df["is_starter"]) & (df["position"] == "GKP")]
    for i in gk_start.index:
        if not df.at[i, "played"] and gk_bench:
            j = gk_bench[0]
            if df.at[j, "played"]:
                df.at[i, "counts"], df.at[i, "subbed_out"] = False, True
                df.at[j, "counts"], df.at[j, "subbed_in"] = True, True
                gk_bench.pop(0)

    # Outfield, in bench order, keeping the formation legal.
    blanks = [
        i for i in df.index
        if df.at[i, "is_starter"] and df.at[i, "position"] != "GKP" and not df.at[i, "played"]
    ]
    for i in blanks:
        for j in list(outfield_bench):
            if not df.at[j, "played"]:
                continue
            trial = [df.at[k, "position"] for k in df.index if df.at[k, "counts"] and k != i]
            trial.append(df.at[j, "position"])
            if _formation_valid(trial):
                df.at[i, "counts"], df.at[i, "subbed_out"] = False, True
                df.at[j, "counts"], df.at[j, "subbed_in"] = True, True
                outfield_bench.remove(j)
                break
    return df


def score_squad(squad_df: pd.DataFrame, actuals: pd.DataFrame,
                captain_multiplier: float = 2.0) -> dict:
    """Realised FPL points for a squad, with captaincy and autosubs."""
    df = squad_df.merge(actuals, on="player_id", how="left")
    df["actual_points"] = df["actual_points"].fillna(0).astype(int)
    df["played"] = df["played"].fillna(False)
    df = apply_autosubs(df)

    df["scored"] = np.where(df["counts"], df["actual_points"], 0)
    cap = df["is_captain"] & df["counts"]
    captain_bonus = int((df.loc[cap, "actual_points"] * (captain_multiplier - 1)).sum())

    # If the captain blanked, FPL hands the armband to the vice-captain.
    cap_row = df[df["is_captain"]]
    cap_played = bool(cap_row["played"].iloc[0]) if len(cap_row) else False
    vice_bonus = 0
    if not cap_played:
        pool = df[(df["counts"]) & (~df["is_captain"])].sort_values("xp", ascending=False)
        if len(pool):
            vice_bonus = int(pool.iloc[0]["actual_points"] * (captain_multiplier - 1))

    base = int(df["scored"].sum())
    total = base + (captain_bonus if cap_played else vice_bonus)
    return {
        "total": total,
        "xi_base": base,
        "captain_bonus": captain_bonus if cap_played else vice_bonus,
        "captain_played": cap_played,
        "captain_used": (cap_row["web_name"].iloc[0] if cap_played and len(cap_row)
                         else "(vice)"),
        "n_autosubs": int(df["subbed_in"].sum()),
        "bench_points": int(df.loc[~df["counts"], "actual_points"].sum()),
        "detail": df,
    }


# --------------------------------------------------------------------------
# Calibration - the part that actually measures the model
# --------------------------------------------------------------------------
def calibration_report(pred: pd.DataFrame, actuals: pd.DataFrame,
                       xp_col: str = "xp", n_bins: int = 10) -> dict:
    """How well do projections track realised points across the whole player pool?

    One squad is a sample of size one and mostly measures luck. Correlation and
    calibration across every player is the real evidence, so this is the output
    worth believing.
    """
    from scipy.stats import pearsonr, spearmanr

    df = pred.merge(actuals, on="player_id", how="inner")
    df["actual_points"] = df["actual_points"].fillna(0)
    x, y = df[xp_col].to_numpy(float), df["actual_points"].to_numpy(float)

    out = {
        "n": len(df),
        "pearson_r": float(pearsonr(x, y)[0]),
        "spearman_r": float(spearmanr(x, y)[0]),
        "mae": float(np.mean(np.abs(x - y))),
        "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
        "mean_pred": float(x.mean()),
        "mean_actual": float(y.mean()),
        "bias": float(x.mean() - y.mean()),
    }

    # Calibration by predicted decile: does bucket k actually score what we said?
    df = df.copy()
    df["_bin"] = pd.qcut(df[xp_col].rank(method="first"), n_bins, labels=False)
    tbl = df.groupby("_bin").agg(
        n=(xp_col, "size"), mean_xp=(xp_col, "mean"),
        mean_actual=("actual_points", "mean"),
    ).reset_index(drop=True)
    tbl["diff"] = tbl["mean_actual"] - tbl["mean_xp"]
    tbl.index = [f"D{i+1}" for i in range(len(tbl))]
    out["by_decile"] = tbl
    return out


def naive_baseline(players: pd.DataFrame, histories: pd.DataFrame,
                   target_event: int) -> pd.Series:
    """Points-per-game to date - the baseline any model must beat.

    If a projection cannot outperform "assume everyone repeats their season
    average", the modelling has added nothing.
    """
    past = histories[histories["event"] < target_event]
    g = past.groupby("player_id").agg(pts=("total_points", "sum"), n=("event", "size"))
    ppg = (g["pts"] / g["n"].clip(lower=1)).rename("baseline_xp")
    return players[["player_id"]].merge(
        ppg, left_on="player_id", right_index=True, how="left"
    ).set_index("player_id")["baseline_xp"].fillna(0.0)


# --------------------------------------------------------------------------
# Component diagnostics - which sub-model was wrong?
# --------------------------------------------------------------------------
def component_diagnostics(pf: pd.DataFrame, client, event: int) -> pd.DataFrame:
    """Compare each projected event total against what actually happened.

    A squad result is one sample and mostly luck. This is the diagnostic that
    tells you *which part* of the model to fix: if expected goals came in fine
    but clean sheets were 40% high, the scoreline model is the problem, not the
    attacking model.
    """
    live = client.live(event)
    thr = {"DEF": 10, "MID": 12, "FWD": 12}
    act = {"goals": 0, "assists": 0, "clean_sheets": 0, "defcon_hits": 0,
           "saves": 0, "bonus": 0, "appearance_pts": 0}
    pos_of = dict(zip(pf["player_id"], pf["position"]))
    for el in live["elements"]:
        s, pos = el["stats"], pos_of.get(el["id"])
        if pos is None:
            continue
        act["goals"] += s["goals_scored"]
        act["assists"] += s["assists"]
        act["saves"] += s["saves"]
        act["bonus"] += s["bonus"]
        if s["minutes"] >= 60:
            act["clean_sheets"] += int(s["clean_sheets"] > 0)
            act["appearance_pts"] += 2
        elif s["minutes"] > 0:
            act["appearance_pts"] += 1
        if pos in thr and s.get("defensive_contribution", 0) >= thr[pos]:
            act["defcon_hits"] += 1

    pred = {
        "goals": pf["exp_goals"].sum(),
        "assists": pf["exp_assists"].sum(),
        "clean_sheets": (pf["p_clean_sheet"] * pf["p_60"]).sum(),
        "defcon_hits": pf["p_defcon"].sum(),
        "saves": pf["exp_saves"].sum(),
        "bonus": pf["xp_bonus"].sum(),
        "appearance_pts": pf["xp_minutes"].sum(),
    }
    rows = []
    for k in pred:
        p, a = float(pred[k]), float(act[k])
        rows.append({"component": k, "predicted": p, "actual": a,
                     "diff": p - a, "pct_error": 100 * (p - a) / a if a else np.nan})
    return pd.DataFrame(rows)


def backtest_gameweek(client, histories: pd.DataFrame, event: int,
                      budget: float = 100.0, verbose: bool = True) -> dict:
    """Full leak-free backtest of one gameweek. Returns everything for inspection."""
    from .fpl import build_fixtures, build_player_fixtures
    from .model import (aggregate_to_players, compute_expected_points,
                        estimate_minutes, fallback_scorelines, team_ratings)
    from .optimise import optimise_free_hit

    boot = client.bootstrap()
    players = reconstruct_players(histories, boot, event)
    fixtures = build_fixtures(client.fixtures(), boot)

    hist = (histories[histories["event"] < event]
            [["player_id", "event", "minutes", "starts"]]
            .rename(columns={"starts": "started"}))
    minutes = estimate_minutes(players, hist, event)

    # Leak guard: nothing from the target gameweek or later may be in scope.
    max_mins = float(players["minutes"].max())
    assert max_mins <= 90.0 * (event - 1) + 1e-6, (
        f"look-ahead detected: max minutes {max_mins} exceeds "
        f"{90 * (event - 1)} available before GW{event}"
    )

    scorelines = fallback_scorelines(fixtures[fixtures["event"] == event],
                                     team_ratings(minutes, fixtures, event))
    pf = compute_expected_points(build_player_fixtures(minutes, fixtures, event), scorelines)
    proj = aggregate_to_players(pf)

    squad = optimise_free_hit(proj, budget=budget)
    actuals = actual_results(client, event)
    scored = score_squad(squad.players, actuals)

    ev = next(e for e in boot["events"] if e["id"] == event)
    if verbose:
        print(f"GW{event}: squad scored {scored['total']} "
              f"(FPL average {ev['average_entry_score']}, highest {ev['highest_score']})")

    return {
        "event": event, "players": proj, "player_fixtures": pf, "squad": squad,
        "actuals": actuals, "scored": scored, "minutes": minutes,
        "fpl_average": ev["average_entry_score"], "fpl_highest": ev["highest_score"],
        "calibration": calibration_report(proj, actuals),
        "components": component_diagnostics(pf, client, event),
    }
