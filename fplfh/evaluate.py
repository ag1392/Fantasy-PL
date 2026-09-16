"""Backtesting against a completed season, with historical market odds.

The live backtest can only test the *statistical* path, because no odds archive
was wired up. That leaves the project's central claim - that market prices beat
a model - completely unverified. This module closes that gap.

### Where the odds come from

football-data.co.uk publishes one CSV per season carrying **actual Betfair
Exchange prices**: match odds and over/under 2.5, which is exactly the pair the
scoreline model fits. Free, no account.

### Opening odds, not closing - this matters

The file has both, and the closing prices are sharper. They are also **look-
ahead**. Their documentation is explicit: standard columns are "collected Friday
afternoons, and on Tuesday afternoons for midweek games", while the closing
columns are taken at kickoff.

FPL deadlines in 2025-26 fell on 20 Saturdays and 12 Fridays, typically 17:30.
Friday-afternoon odds precede those. A Sunday or Monday fixture's *closing* odds
are taken a day or two **after** the deadline, and carry team news no manager
could have had.

The cost of doing this correctly is small: mean change in P(home) from open to
close is 0.021, correlation 0.989.

One residual caveat, noted rather than engineered around: for Friday 17:30
deadlines the collection time is close to the deadline and could occasionally
sit the wrong side of it. The drift over a few hours is far below 0.021.
"""
from __future__ import annotations

import io as _io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import CACHE_DIR, ELEMENT_TYPE_TO_POS
from .model.scoreline import ScorelineDistribution, fit_to_markets
from .naming import team_key

FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
VAASTAV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
           "master/data/{season}/gws/merged_gw.csv")

_POS = {"GK": "GKP", "GKP": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}


def _season_code(season: str) -> str:
    """'2025-26' -> '2526', the form football-data.co.uk uses."""
    a, b = season.split("-")
    return a[2:] + b[-2:]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_odds(season: str = "2025-26", refresh: bool = False,
              closing: bool = False) -> pd.DataFrame:
    """Historical Betfair Exchange prices, de-vigged.

    ``closing=True`` is available for comparison but should not be used to
    score a backtest - see the module docstring.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"odds_{season}.csv"
    if refresh or not path.exists():
        url = FOOTBALL_DATA.format(code=_season_code(season))
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    raw = pd.read_csv(_io.BytesIO(path.read_bytes()), encoding="utf-8-sig")

    pre = "BFEC" if closing else "BFE"
    cols = {"h": f"{pre}H", "d": f"{pre}D", "a": f"{pre}A",
            "over": f"{pre}>2.5", "under": f"{pre}<2.5"}
    missing = [c for c in cols.values() if c not in raw.columns]
    if missing:
        raise KeyError(f"{season} file is missing {missing}")

    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce"),
        "home": raw["HomeTeam"], "away": raw["AwayTeam"],
        "fthg": raw.get("FTHG"), "ftag": raw.get("FTAG"),
    })
    # 1X2 is mutually exclusive and exhaustive, so normalising removes the
    # residual spread. Same for the two-way over/under book.
    inv = raw[[cols["h"], cols["d"], cols["a"]]].rdiv(1.0)
    tot = inv.sum(axis=1)
    out["p_home"] = inv[cols["h"]] / tot
    out["p_draw"] = inv[cols["d"]] / tot
    out["p_away"] = inv[cols["a"]] / tot

    ou = raw[[cols["over"], cols["under"]]].rdiv(1.0)
    out["p_over25"] = ou[cols["over"]] / ou.sum(axis=1)
    out["source"] = "betfair_exchange_" + ("closing" if closing else "preclose")
    return out.dropna(subset=["p_home", "p_draw", "p_away"]).reset_index(drop=True)


def load_history(season: str = "2025-26", refresh: bool = False) -> pd.DataFrame:
    """Per-player, per-gameweek FPL data for a completed season."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"history_{season}.csv"
    if refresh or not path.exists():
        resp = requests.get(VAASTAV.format(season=season), timeout=180)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    df = pd.read_csv(path)
    df["position"] = df["position"].map(_POS).fillna(df["position"])
    for c in ("expected_goals", "expected_assists", "expected_goals_conceded"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def build_historical_fixtures(history: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Rebuild the season's fixture list from per-player rows.

    Each fixture appears twice - once per side - so the pair of (team,
    opponent_team) rows identifies both clubs and recovers the id-to-name map
    that the raw file does not carry.
    """
    id_to_name: dict[int, str] = {}
    for _, grp in history.groupby("fixture"):
        sides = grp.drop_duplicates("team")[["team", "opponent_team"]]
        if len(sides) != 2:
            continue
        (t1, o1), (t2, o2) = sides.itertuples(index=False)
        # t1's opponent id belongs to t2, and vice versa.
        id_to_name.setdefault(int(o1), t2)
        id_to_name.setdefault(int(o2), t1)

    rows = []
    for fid, grp in history.groupby("fixture"):
        home = grp[grp.was_home]
        away = grp[~grp.was_home]
        if home.empty or away.empty:
            continue
        h, a = home.iloc[0], away.iloc[0]
        rows.append({
            "fixture_id": int(fid), "event": int(h["GW"]),
            "kickoff": pd.to_datetime(h["kickoff_time"], utc=True, errors="coerce"),
            "team_h_short": h["team"], "team_a_short": a["team"],
            "team_h_name": h["team"], "team_a_name": a["team"],
            "team_h": int(a["opponent_team"]), "team_a": int(h["opponent_team"]),
            "finished": True,
            "home_goals": h.get("team_h_score"), "away_goals": h.get("team_a_score"),
        })
    fixtures = pd.DataFrame(rows).sort_values(["event", "kickoff"]).reset_index(drop=True)
    return fixtures, id_to_name


def match_odds_to_fixtures(fixtures: pd.DataFrame, odds: pd.DataFrame,
                           tolerance_days: int = 2) -> pd.DataFrame:
    """Join the odds file to the fixture list on team names and date."""
    odds = odds.copy()
    odds["_h"] = odds["home"].map(lambda s: frozenset(team_key(s)))
    odds["_a"] = odds["away"].map(lambda s: frozenset(team_key(s)))

    out = []
    for _, fx in fixtures.iterrows():
        hk, ak = team_key(fx["team_h_name"]), team_key(fx["team_a_name"])
        ko = pd.to_datetime(fx["kickoff"], utc=True, errors="coerce")
        best = None
        for _, o in odds.iterrows():
            if not (hk & o["_h"]) or not (ak & o["_a"]):
                continue
            gap = abs((pd.Timestamp(o["date"], tz="UTC") - ko).days) if pd.notna(ko) else 0
            if gap > tolerance_days:
                continue
            if best is None or gap < best[0]:
                best = (gap, o)
        if best is not None:
            o = best[1]
            out.append({"fixture_id": int(fx["fixture_id"]),
                        "p_home": o["p_home"], "p_draw": o["p_draw"],
                        "p_away": o["p_away"], "p_over25": o["p_over25"]})
    return pd.DataFrame(out)


def scorelines_from_odds(joined: pd.DataFrame) -> dict[int, ScorelineDistribution]:
    """Fit a scoreline model per fixture from historical market probabilities."""
    out: dict[int, ScorelineDistribution] = {}
    for _, r in joined.iterrows():
        lines = {2.5: float(r["p_over25"])} if pd.notna(r.get("p_over25")) else None
        try:
            out[int(r["fixture_id"])] = fit_to_markets(
                p_home=float(r["p_home"]), p_draw=float(r["p_draw"]),
                p_away=float(r["p_away"]), over_lines=lines,
                source="historical:betfair_exchange",
            )
        except ValueError:
            continue
    return out


# --------------------------------------------------------------------------
# Player state, without look-ahead
# --------------------------------------------------------------------------
def reconstruct_players(history: pd.DataFrame, id_to_name: dict[int, str],
                        upto_gw: int) -> pd.DataFrame:
    """Player state as it stood before ``upto_gw``, from prior gameweeks only.

    Column-compatible with ``fpl.schema.build_players`` so the rest of the
    pipeline runs unchanged. Availability flags are unavailable historically, so
    every player is treated as fit - the same deliberate handicap the live
    backtest applies.
    """
    name_to_id = {v: k for k, v in id_to_name.items()}
    past = history[history["GW"] < upto_gw]
    if past.empty:
        raise ValueError(f"no data before gameweek {upto_gw}")

    sums = ["minutes", "starts", "goals_scored", "assists", "yellow_cards",
            "red_cards", "saves", "bonus", "total_points", "defensive_contribution",
            "expected_goals", "expected_assists", "expected_goals_conceded"]
    agg = past.groupby("element")[sums].sum()
    meta = past.sort_values("GW").groupby("element").agg(
        web_name=("name", "last"), position=("position", "last"), team=("team", "last"))
    # A player in a double gameweek has two rows, so take one price per player
    # rather than setting a non-unique index. The gameweek's own `value` is the
    # price at its deadline, which is known when the pick is made.
    price = (history[history["GW"] == upto_gw]
             .groupby("element")["value"].last())
    if price.empty:
        price = past.sort_values("GW").groupby("element")["value"].last()

    df = agg.join(meta).reset_index().rename(columns={"element": "player_id"})
    last_seen = past.sort_values("GW").groupby("element")["value"].last()
    df["cost_tenths"] = (df["player_id"].map(price)
                         .fillna(df["player_id"].map(last_seen))
                         .fillna(45).astype(int))
    df["price"] = df["cost_tenths"] / 10.0
    df["team_id"] = df["team"].map(name_to_id).fillna(-1).astype(int)
    df["team_name"] = df["team"]

    n90 = (df["minutes"] / 90.0).replace(0, np.nan)
    df["nineties"] = df["minutes"] / 90.0
    for out_col, src in (("xg90", "expected_goals"), ("xa90", "expected_assists"),
                         ("xgc90", "expected_goals_conceded"),
                         ("dc90", "defensive_contribution"), ("saves90", "saves"),
                         ("yc90", "yellow_cards")):
        df[out_col] = (df[src] / n90).fillna(0.0)
    df = df.rename(columns={"expected_goals": "xg_total", "expected_assists": "xa_total",
                            "defensive_contribution": "dc_total", "saves": "saves_total",
                            "goals_scored": "goals"})

    # Not available historically; the penalty-miss term is worth hundredths.
    df["status"], df["news"], df["chance_of_playing"] = "a", "", None
    df["selected_by_percent"] = 0.0
    for c in ("pens_order", "corners_order", "fk_order"):
        df[c] = np.nan
    df["is_pen_taker"] = False
    df["full_name"] = df["web_name"]
    return df


def actual_points(history: pd.DataFrame, gw: int) -> pd.DataFrame:
    """What actually happened in one gameweek."""
    d = history[history["GW"] == gw]
    return pd.DataFrame({
        "player_id": d["element"], "actual_points": d["total_points"],
        "actual_minutes": d["minutes"], "actual_goals": d["goals_scored"],
        "actual_assists": d["assists"], "actual_saves": d["saves"],
        "actual_bonus": d["bonus"], "actual_cs": (d["goals_conceded"] == 0) & (d["minutes"] >= 60),
        "actual_dc": d["defensive_contribution"], "actual_yc": d["yellow_cards"],
        "played": d["minutes"] > 0,
    }).groupby("player_id", as_index=False).sum()


# --------------------------------------------------------------------------
# Running a season
# --------------------------------------------------------------------------
def evaluate_season(season: str = "2025-26", gameweeks=None, min_gw: int = 3,
                    verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project every gameweek of a completed season and keep what happened.

    Returns ``(per_player, per_fixture)``: one row per player-gameweek with the
    projection alongside the realised outcome, and one row per fixture with the
    scoreline model alongside the final score.

    Scorelines come from the market where a fixture was priced and from the
    ratings model where it was not - exactly what the live tool does, so the
    numbers describe the tool as it is actually used.

    Gameweeks before ``min_gw`` are skipped: with one or two gameweeks of
    history every rate is still essentially its prior, so those rows measure the
    prior rather than the model.
    """
    from .fpl import build_player_fixtures
    from .model import (compute_expected_points, estimate_minutes,
                        fallback_scorelines, team_ratings)

    history = load_history(season)
    fixtures, id_to_name = build_historical_fixtures(history)
    market = scorelines_from_odds(
        match_odds_to_fixtures(fixtures, load_odds(season)))
    if verbose:
        print(f"{season}: {len(fixtures)} fixtures, {len(market)} priced by the market")

    gws = sorted(g for g in history["GW"].dropna().unique() if g >= min_gw)
    if gameweeks is not None:
        gws = [g for g in gws if g in set(gameweeks)]

    player_rows, fixture_rows = [], []
    for gw in gws:
        gw_fx = fixtures[fixtures["event"] == gw]
        if gw_fx.empty:
            continue

        # Everything is rebuilt from gameweeks strictly before this one. The
        # obvious shortcut - reading season-to-date totals - would let a
        # player's expected goals include the gameweek being predicted.
        players = reconstruct_players(history, id_to_name, gw)
        mins_hist = (history[history["GW"] < gw][["element", "GW", "minutes", "starts"]]
                     .rename(columns={"element": "player_id", "GW": "event",
                                      "starts": "started"}))
        # Manual overrides are keyed to this season's player ids and would be
        # meaningless applied to a past one.
        minutes = estimate_minutes(players, mins_hist, gw, apply_overrides=False)

        scorelines = fallback_scorelines(gw_fx, team_ratings(players, fixtures, gw))
        scorelines.update({int(f): market[int(f)] for f in gw_fx["fixture_id"]
                           if int(f) in market})

        pf = compute_expected_points(
            build_player_fixtures(minutes, fixtures, gw), scorelines)
        agg = pf.groupby("player_id", as_index=False).agg(
            {**{c: "sum" for c in _SUM_COLS if c in pf},
             **{c: "max" for c in _MAX_COLS if c in pf},
             **{c: "first" for c in _FIRST_COLS if c in pf}})
        agg["event"] = gw
        player_rows.append(agg.merge(actual_points(history, gw),
                                     on="player_id", how="left"))

        for _, fx in gw_fx.iterrows():
            d = scorelines.get(int(fx["fixture_id"]))
            if d is None:
                continue
            fixture_rows.append({
                "event": gw, "fixture_id": int(fx["fixture_id"]),
                "home": fx["team_h_name"], "away": fx["team_a_name"],
                "xg_home": d.expected_goals(True), "xg_away": d.expected_goals(False),
                "cs_home": d.p_clean_sheet(True), "cs_away": d.p_clean_sheet(False),
                "goals_home": fx["home_goals"], "goals_away": fx["away_goals"],
                # The full correct-score matrix, so the *shape* of the forecast
                # can be checked and not just its mean.
                "joint": d.joint,
            })

        if verbose and gw % 5 == 0:
            print(f"  gameweek {gw}")

    per_player = pd.concat(player_rows, ignore_index=True)
    for c in per_player.columns:
        if c.startswith("actual_") or c == "played":
            per_player[c] = per_player[c].fillna(0)
    return per_player, pd.DataFrame(fixture_rows)


_SUM_COLS = ["xp", "xp_minutes", "xp_goals", "xp_assists", "xp_clean_sheet",
             "xp_conceded", "xp_defcon", "xp_saves", "xp_cards", "xp_bonus",
             "xp_pens", "exp_goals", "exp_assists", "exp_saves", "p_defcon",
             "p_clean_sheet_scored"]
_MAX_COLS = ["p_clean_sheet", "p_60", "p_start", "xmins"]
_FIRST_COLS = ["web_name", "team", "team_id", "position", "price"]


# --------------------------------------------------------------------------
# Scoring each model against what happened
# --------------------------------------------------------------------------
# Every component paired with the realised quantity it is a prediction *of*.
# Deliberately in natural units - goals, clean sheets, hits, minutes - rather
# than points: a component can be right about points for the wrong reason, and
# natural units are what make an error interpretable.
COMPONENT_PAIRS = {
    "minutes":      ("xmins", "actual_minutes"),
    "appearances":  ("p_play_eff", "played"),
    "goals":        ("exp_goals", "actual_goals"),
    "assists":      ("exp_assists", "actual_assists"),
    "clean sheets": ("p_clean_sheet_eff", "actual_cs"),
    "defcon hits":  ("p_defcon", "actual_dc_hit"),
    "saves":        ("exp_saves", "actual_saves"),
    "yellow cards": ("exp_yc", "actual_yc"),
    "bonus points": ("xp_bonus", "actual_bonus"),
}


def _prepare(per_player: pd.DataFrame) -> pd.DataFrame:
    """Derive the realised counterparts the raw frame does not carry."""
    from .config import params

    d = per_player.copy()
    sc = params()["scoring"]

    # A DefCon point needs a position-specific number of defensive actions.
    thr = sc["defensive_contribution"]["thresholds"]
    limit = d["position"].map(lambda p: float(thr.get(p, np.inf)))
    d["actual_dc_hit"] = (d["actual_dc"].fillna(0) >= limit).astype(float)

    # Clean sheets pay only to GKP/DEF/MID, and only with 60 minutes. The
    # comparable prediction is the probability the model actually scores on.
    #
    # The *realised* side has to be restricted the same way. A forward whose
    # team shuts the opposition out has kept a clean sheet in the everyday sense
    # but scores nothing for it, so counting those inflates the denominator and
    # makes the model look like it is under-predicting when it is correctly
    # predicting zero.
    scores_cs = d["position"].isin(["GKP", "DEF", "MID"])
    d["p_clean_sheet_eff"] = np.where(
        scores_cs,
        d.get("p_clean_sheet_scored", d["p_clean_sheet"] * d["p_60"]).fillna(0), 0.0)
    d["actual_cs"] = np.where(scores_cs, d["actual_cs"].fillna(0), 0.0).astype(float)

    # Cards and appearances are carried as points rather than counts; both are
    # linear in the underlying quantity, so back the count out.
    yc_pts = abs(float(sc["yellow_cards"]))
    d["exp_yc"] = d["xp_cards"].abs() / yc_pts if yc_pts else 0.0

    short = float(sc["minutes"]["short_play_pts"])
    long = float(sc["minutes"]["long_play_pts"])
    d["p_play_eff"] = ((d["xp_minutes"] - d["p_60"].fillna(0) * (long - short))
                       / short if short else 0.0)
    d["played"] = d["played"].fillna(0).astype(float)
    return d


def score_components(per_player: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Each model on its own, predicted against realised.

    One row per component: what the model expected across the season, what
    actually happened, and the gap. ``per_player_pred`` and ``per_player_actual``
    are the same figures as averages, which is usually the easier read.

    This is the table to start from. A total can look healthy while the parts
    are wrong in opposite directions - an over-predicted clean sheet cancelling
    an under-predicted DefCon - and only a component view shows that.
    """
    d = _prepare(per_player)
    groups = [("all", d)] if by is None else list(d.groupby(by, observed=True))

    rows = []
    for key, g in groups:
        n = len(g)
        for name, (pred, act) in COMPONENT_PAIRS.items():
            if pred not in g or act not in g:
                continue
            p = float(g[pred].fillna(0).sum())
            a = float(g[act].fillna(0).sum())
            row = {"component": name, "predicted": p, "actual": a,
                   "diff": p - a, "pct_error": 100 * (p - a) / a if a else np.nan,
                   "per_player_pred": p / n if n else np.nan,
                   "per_player_actual": a / n if n else np.nan}
            if by is not None:
                row = {by: key, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def score_scorelines(per_fixture: pd.DataFrame) -> pd.DataFrame:
    """The scoreline model on its own, before any player is involved.

    Expected goals against goals actually scored, per match. This is the model
    the market feeds directly, so it is the cleanest read on whether the market
    prices are doing their job.
    """
    d = per_fixture.dropna(subset=["goals_home", "goals_away"]).copy()
    if d.empty:
        return pd.DataFrame()

    h = d["goals_home"].astype(float)
    a = d["goals_away"].astype(float)
    pairs = [
        ("home goals", d["xg_home"], h),
        ("away goals", d["xg_away"], a),
        ("total goals", d["xg_home"] + d["xg_away"], h + a),
        ("clean sheets", pd.concat([d["cs_home"], d["cs_away"]]),
         pd.concat([(a == 0).astype(float), (h == 0).astype(float)])),
    ]
    rows = []
    for name, pred, act in pairs:
        p, o = float(pred.mean()), float(act.mean())
        rows.append({"quantity": name, "per_match_pred": p, "per_match_actual": o,
                     "diff": p - o, "pct_error": 100 * (p - o) / o if o else np.nan})
    return pd.DataFrame(rows)


def score_distribution(per_fixture: pd.DataFrame, max_goals: int = 4) -> pd.DataFrame:
    """Predicted against observed frequency of each correct score.

    Matching the mean is a weak test. A model can predict exactly the right
    number of goals per match while getting the *shape* wrong - too many 1-1s
    and not enough 0-0s and 4-2s - and every clean sheet, and therefore every
    defender, is priced off that shape rather than off the mean.

    The scoreline model is a Dixon-Coles bivariate Poisson, whose whole purpose
    is the low-score correction: independent Poissons understate 0-0 and 1-1 and
    overstate 1-0 and 0-1. This is the table that says whether the correction is
    doing its job.

    Scores above ``max_goals`` are pooled into an "other" row rather than shown
    individually, since each is individually rare.
    """
    d = per_fixture.dropna(subset=["goals_home", "goals_away"]).copy()
    if d.empty or "joint" not in d:
        return pd.DataFrame()

    n = len(d)
    grid = np.zeros_like(np.asarray(d["joint"].iloc[0], dtype=float))
    for j in d["joint"]:
        grid = grid + np.asarray(j, dtype=float)

    h = d["goals_home"].astype(int).to_numpy()
    a = d["goals_away"].astype(int).to_numpy()

    rows, seen = [], np.zeros_like(grid, dtype=bool)
    for i in range(min(max_goals + 1, grid.shape[0])):
        for k in range(min(max_goals + 1, grid.shape[1])):
            seen[i, k] = True
            obs = int(((h == i) & (a == k)).sum())
            rows.append({"score": f"{i}-{k}", "predicted": grid[i, k],
                         "observed": float(obs)})
    # Everything outside the displayed grid, pooled.
    pred_other = float(grid[~seen].sum())
    obs_other = int(n - sum(r["observed"] for r in rows))
    rows.append({"score": f">{max_goals} either side", "predicted": pred_other,
                 "observed": float(obs_other)})

    out = pd.DataFrame(rows)
    out["pred_pct"] = 100 * out["predicted"] / n
    out["obs_pct"] = 100 * out["observed"] / n
    out["diff_pct"] = out["pred_pct"] - out["obs_pct"]
    return out.sort_values("observed", ascending=False).reset_index(drop=True)


def goal_count_distribution(per_fixture: pd.DataFrame, max_goals: int = 6) -> pd.DataFrame:
    """Predicted against observed frequency of total goals in a match.

    The one-dimensional view of the same question, and the one the over/under
    market prices directly - so a mismatch here points at the market input
    rather than at the correlation structure.
    """
    d = per_fixture.dropna(subset=["goals_home", "goals_away"]).copy()
    if d.empty or "joint" not in d:
        return pd.DataFrame()

    n = len(d)
    grid = np.zeros_like(np.asarray(d["joint"].iloc[0], dtype=float))
    for j in d["joint"]:
        grid = grid + np.asarray(j, dtype=float)
    totals = np.add.outer(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
    actual = (d["goals_home"] + d["goals_away"]).astype(int).to_numpy()

    rows = []
    for t in range(max_goals + 1):
        rows.append({"total_goals": str(t), "predicted": float(grid[totals == t].sum()),
                     "observed": float((actual == t).sum())})
    rows.append({"total_goals": f"{max_goals + 1}+",
                 "predicted": float(grid[totals > max_goals].sum()),
                 "observed": float((actual > max_goals).sum())})

    out = pd.DataFrame(rows)
    out["pred_pct"] = 100 * out["predicted"] / n
    out["obs_pct"] = 100 * out["observed"] / n
    out["diff_pct"] = out["pred_pct"] - out["obs_pct"]
    return out


# --------------------------------------------------------------------------
# Look-ahead audit
# --------------------------------------------------------------------------
def audit_look_ahead(history: pd.DataFrame, fixtures: pd.DataFrame,
                     id_to_name: dict[int, str], gameweeks=(4, 10, 20, 30, 38)
                     ) -> pd.DataFrame:
    """Prove the reconstruction contains nothing from the gameweek it predicts.

    The obvious check - no player may hold more than 90 x (gw - 1) minutes - is
    **wrong**, and quietly so. Double gameweeks mean a team can play more
    matches than gameweeks: before GW30 in 2025-26, Arsenal had played 30
    matches in 29 gameweeks, so their keeper held 2,700 minutes against a
    supposed ceiling of 2,610. That check fires a false alarm rather than
    catching anything.

    What is tested instead:

    1. **Exact reconstruction.** Every cumulative stat must equal the sum over
       gameweeks strictly before the target, to floating-point tolerance. This
       is airtight: if any future row contributed, the totals would not match.
    2. **A fixture-aware ceiling.** Minutes cannot exceed 90 x the number of
       matches the player's team actually played beforehand.
    """
    stats = {"minutes": "minutes", "goals": "goals_scored", "xg_total": "expected_goals",
             "xa_total": "expected_assists", "dc_total": "defensive_contribution",
             "saves_total": "saves"}
    rows = []
    for gw in gameweeks:
        past = history[history["GW"] < gw]
        if past.empty:
            continue
        p = reconstruct_players(history, id_to_name, gw).set_index("player_id")
        prior = past.groupby("element")[list(stats.values())].sum()

        worst = 0.0
        for got, want in stats.items():
            delta = (p[got] - prior[want].reindex(p.index).fillna(0.0)).abs().max()
            worst = max(worst, float(delta))

        # Matches actually played by each team before this gameweek.
        played = pd.concat([
            fixtures[fixtures["event"] < gw]["team_h_name"],
            fixtures[fixtures["event"] < gw]["team_a_name"],
        ]).value_counts()
        ceiling = p["team"].map(played).fillna(0) * 90.0
        over = int((p["minutes"] > ceiling + 1e-9).sum())

        rows.append({"gameweek": gw, "players": len(p),
                     "max_abs_stat_error": worst,
                     "players_over_fixture_ceiling": over,
                     "naive_ceiling_would_flag": int(
                         (p["minutes"] > 90 * (gw - 1) + 1e-9).sum()),
                     "passes": worst < 1e-6 and over == 0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Free Hit backtest
# --------------------------------------------------------------------------
def template_xi(history: pd.DataFrame, gw: int, formation_min=(1, 3, 2, 1),
                max_per_team: int = 3) -> pd.DataFrame:
    """The most-owned legal XI for a gameweek - a stand-in for the crowd.

    FPL's published weekly average is not available for a past season: the API
    carries ``average_entry_score`` only for the season in progress, and no
    archive republishes it. What *is* available is ``selected`` - the number of
    managers owning each player - so the crowd's team can be rebuilt directly.

    This picks the most-owned eleven subject to the real constraints (one
    keeper, at least three defenders, two midfielders and a forward, at most
    three per club) and captains the most-owned player. It is a proxy, and it
    will run slightly **above** the true average: it never carries an injured
    player, never takes a hit, and always captains the popular pick.
    """
    d = history[history["GW"] == gw].copy()
    if d.empty or "selected" not in d:
        return pd.DataFrame()
    d = d.sort_values("selected", ascending=False).drop_duplicates("element")

    need = dict(zip(("GKP", "DEF", "MID", "FWD"), formation_min))
    picked, counts, clubs = [], {k: 0 for k in need}, {}
    for _, r in d.iterrows():
        pos, club = r["position"], r["team"]
        if clubs.get(club, 0) >= max_per_team:
            continue
        cap = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}[pos]
        if counts[pos] >= cap:
            continue
        # Leave room for the positions still short of their minimum.
        short = sum(max(0, need[k] - counts[k]) for k in need if k != pos)
        if len(picked) + 1 + short > 11:
            continue
        picked.append(r)
        counts[pos] += 1
        clubs[club] = clubs.get(club, 0) + 1
        if len(picked) == 11:
            break
    return pd.DataFrame(picked)


def backtest_free_hit(season: str = "2025-26", start_gw: int = 4, end_gw: int = 38,
                      budget: float = 100.0, verbose: bool = True) -> pd.DataFrame:
    """What the Free Hit optimiser would have fielded, week by week, and scored.

    For each gameweek: rebuild player state from earlier gameweeks only, project,
    solve the squad, then look up what those players actually scored.

    Three reference points are returned alongside it - the crowd's most-owned
    XI, a perfect-hindsight XI under the same budget, and the mean score of all
    players who featured - because a raw points total means nothing without
    knowing what was achievable and what was typical.
    """
    from .fpl import build_player_fixtures
    from .model import (aggregate_to_players, compute_expected_points,
                        estimate_minutes, fallback_scorelines, team_ratings)
    from .optimise import optimise_free_hit

    history = load_history(season)
    fixtures, id_to_name = build_historical_fixtures(history)
    market = scorelines_from_odds(
        match_odds_to_fixtures(fixtures, load_odds(season)))

    rows = []
    for gw in range(start_gw, end_gw + 1):
        gw_fx = fixtures[fixtures["event"] == gw]
        if gw_fx.empty:
            continue

        players = reconstruct_players(history, id_to_name, gw)
        mins_hist = (history[history["GW"] < gw][["element", "GW", "minutes", "starts"]]
                     .rename(columns={"element": "player_id", "GW": "event",
                                      "starts": "started"}))
        minutes = estimate_minutes(players, mins_hist, gw, apply_overrides=False)

        scorelines = fallback_scorelines(gw_fx, team_ratings(players, fixtures, gw))
        scorelines.update({int(f): market[int(f)] for f in gw_fx["fixture_id"]
                           if int(f) in market})
        pf = compute_expected_points(
            build_player_fixtures(minutes, fixtures, gw), scorelines)
        pool = aggregate_to_players(pf)

        squad = optimise_free_hit(pool, budget=budget)
        if squad is None:
            continue

        actual = actual_points(history, gw).set_index("player_id")["actual_points"]
        sq = squad.players.copy()
        sq["actual"] = sq["player_id"].map(actual).fillna(0.0)
        xi = sq[sq["is_starter"]]
        cap_pts = float(xi.loc[xi["is_captain"], "actual"].sum())

        # Perfect hindsight under the same budget: the same optimiser handed the
        # answers. The gap to this is the cost of not knowing the future, which
        # is most of the gap - it is not a target, it is a ceiling.
        hind = pool.copy()
        hind["xp"] = hind["player_id"].map(actual).fillna(0.0)
        best = optimise_free_hit(hind, budget=budget)

        tmpl = template_xi(history, gw)
        tmpl_pts = (float(tmpl["total_points"].sum()
                          + tmpl["total_points"].iloc[0]) if len(tmpl) else np.nan)

        played = history[(history["GW"] == gw) & (history["minutes"] > 0)]

        rows.append({
            "event": gw,
            "xp": squad.starting_xp,
            "actual": float(xi["actual"].sum()) + cap_pts,
            "captain": squad.captain,
            "captain_pts": cap_pts,
            "formation": squad.formation,
            "cost": squad.cost,
            "template": tmpl_pts,
            "hindsight": (float(best.players[best.players["is_starter"]]["xp"].sum())
                          + float(best.players[
                              best.players["is_captain"]]["xp"].sum())
                          if best is not None else np.nan),
            "mean_player_pts": float(played["total_points"].mean()),
        })
        if verbose:
            r = rows[-1]
            print(f"  GW{gw:<3d} xP {r['xp']:5.1f}   scored {r['actual']:5.0f}"
                  f"   template {r['template']:5.0f}"
                  f"   ceiling {r['hindsight']:5.0f}   (C) {r['captain'][:16]}")

    out = pd.DataFrame(rows)
    if len(out):
        out["vs_template"] = out["actual"] - out["template"]
    return out
