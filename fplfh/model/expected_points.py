"""Assemble expected points per (player, fixture), then per player.

Every component column is kept alongside the total, and every fixture carries a
``xp_source`` tag naming where its scoreline came from. That matters more than
it might seem: the interesting question is rarely "what is this player worth",
it is "which part of this number am I actually confident in". A projection
built on a match-odds book with millions matched deserves more trust than one
resting on a fallback ratings model, and the output should say which it is.

Because the unit of work is (player, fixture), double gameweeks sum naturally
and blanks fall out at zero without any special-casing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import params
from . import events as ev
from .scoreline import ScorelineDistribution

COMPONENTS = [
    "xp_minutes",
    "xp_goals",
    "xp_assists",
    "xp_clean_sheet",
    "xp_conceded",
    "xp_defcon",
    "xp_saves",
    "xp_cards",
    "xp_bonus",
    "xp_pens",
]


def compute_expected_points(
    player_fixtures: pd.DataFrame,
    scorelines: dict[int, ScorelineDistribution],
    anytime_goal_probs: dict[tuple[int, int], float] | None = None,
) -> pd.DataFrame:
    """Expected points for each (player, fixture) row.

    ``scorelines`` maps fixture_id -> fitted distribution.
    ``anytime_goal_probs`` optionally maps (fixture_id, player_id) -> de-vigged
    P(player scores), used in preference to the xG-share fallback.
    """
    sc = params()["scoring"]
    df = player_fixtures.copy()

    # ---- team-level quantities, read off the scoreline distribution --------
    def _lookup(row, fn, default=np.nan):
        dist = scorelines.get(row["fixture_id"])
        return default if dist is None else fn(dist, bool(row["is_home"]))

    df["team_lambda"] = df.apply(lambda r: _lookup(r, lambda d, h: d.expected_goals(h)), axis=1)
    df["opp_lambda"] = df.apply(lambda r: _lookup(r, lambda d, h: d.expected_conceded(h)), axis=1)
    df["p_clean_sheet"] = df.apply(lambda r: _lookup(r, lambda d, h: d.p_clean_sheet(h)), axis=1)
    df["exp_conceded_units"] = df.apply(
        lambda r: _lookup(
            r,
            lambda d, h: d.expected_conceded_penalty_units(
                h, int(sc["goals_conceded"]["per_n_conceded"])
            ),
        ),
        axis=1,
    )
    df["xp_source"] = df["fixture_id"].map(
        {fid: d.source for fid, d in scorelines.items()}
    ).fillna("missing")
    df["closeness"] = df["fixture_id"].map(
        {
            fid: ev.match_closeness(d.p_win(True), d.p_draw(), d.p_win(False))
            for fid, d in scorelines.items()
        }
    ).fillna(1.0)

    # Rows with no priced fixture cannot be projected; drop them to zero rather
    # than silently producing NaN totals downstream.
    unpriced = df["team_lambda"].isna()
    df.loc[unpriced, ["team_lambda", "opp_lambda", "p_clean_sheet", "exp_conceded_units"]] = 0.0

    # ---- appearance --------------------------------------------------------
    df["xp_minutes"] = df["p_play"] * float(sc["minutes"]["short_play_pts"]) + df["p_60"] * (
        float(sc["minutes"]["long_play_pts"]) - float(sc["minutes"]["short_play_pts"])
    )

    # ---- shrink noisy per-90 rates before they drive anything --------------
    # Must happen before the goal/assist shares are built: the share model
    # normalises to the team total, so an inflated individual rate does not
    # raise the team's expected goals, it *steals* them from team-mates.
    prior_n90 = float(params()["rates"]["prior_nineties"])
    if "price_bucket" not in df.columns:
        df["price_bucket"] = pd.cut(
            df["price"], bins=[0, 4.4, 4.9, 5.4, 6.4, 7.9, 30.0],
            labels=["<=4.4", "4.5-4.9", "5.0-5.4", "5.5-6.4", "6.5-7.9", "8.0+"],
        )
    df["xg90_shrunk"] = ev.shrink_rate_grouped(df, "xg90", prior_n90)
    df["xa90_shrunk"] = ev.shrink_rate_grouped(df, "xa90", prior_n90)

    # ---- goals -------------------------------------------------------------
    df["goal_weight"] = np.nan_to_num(df["xg90_shrunk"]) * (df["xmins"] / 90.0)
    df["exp_goals"] = 0.0
    df["goals_source"] = "model:xg_share"

    for (fid, team_id), grp in df.groupby(["fixture_id", "team_id"], sort=False):
        lam = float(grp["team_lambda"].iloc[0])
        if lam <= 0:
            continue
        idx = grp.index
        market = None
        if anytime_goal_probs:
            probs = np.array(
                [anytime_goal_probs.get((fid, int(pid)), np.nan) for pid in grp["player_id"]]
            )
            # Use the market only when it covers the bulk of the team's threat,
            # otherwise its shape is not representative of the squad.
            covered = np.isfinite(probs)
            if covered.sum() >= 5 and grp.loc[idx[covered], "goal_weight"].sum() >= 0.6 * max(
                grp["goal_weight"].sum(), 1e-9
            ):
                filled = np.where(covered, np.nan_to_num(probs), 0.0)
                market = ev.goals_from_anytime_market(filled, lam)
        if market is not None:
            df.loc[idx, "exp_goals"] = market
            df.loc[idx, "goals_source"] = "market:anytime_goalscorer->team_total"
        else:
            df.loc[idx, "exp_goals"] = ev.distribute_team_goals(grp["goal_weight"].values, lam)

    goal_pts = df["position"].map(sc["goals_scored"]).astype(float)
    df["xp_goals"] = df["exp_goals"] * goal_pts

    # ---- assists -----------------------------------------------------------
    df["assist_weight"] = np.nan_to_num(df["xa90_shrunk"]) * (df["xmins"] / 90.0)
    df["exp_assists"] = 0.0
    for (fid, team_id), grp in df.groupby(["fixture_id", "team_id"], sort=False):
        lam = float(grp["team_lambda"].iloc[0])
        if lam <= 0:
            continue
        df.loc[grp.index, "exp_assists"] = ev.distribute_team_assists(
            grp["assist_weight"].values, lam
        )
    df["xp_assists"] = df["exp_assists"] * float(sc["assists"])

    # ---- clean sheets and goals conceded -----------------------------------
    cs_pts = df["position"].map(sc["clean_sheets"]).astype(float)
    # A clean sheet requires 60+ minutes, so it is gated on p_60, not p_play -
    # and it is awarded for conceding nothing *while on the pitch*, which is a
    # shorter window than the match. See events.on_pitch_clean_sheet.
    df["p_clean_sheet_scored"] = ev.on_pitch_clean_sheet(
        df["p_clean_sheet"], df["p_60"]
    )
    df["xp_clean_sheet"] = df["p_clean_sheet_scored"] * cs_pts

    conceded_pts = df["position"].map(sc["goals_conceded"]["pts"]).fillna(0.0).astype(float)
    # Conceded deductions accrue while on the pitch; scale by minutes share.
    df["xp_conceded"] = (
        df["exp_conceded_units"] * conceded_pts * np.clip(df["xmins"] / 90.0, 0.0, 1.0)
    )

    # ---- defensive contribution -------------------------------------------
    dcfg = params()["defcon"]
    df["xp_defcon"] = 0.0
    df["p_defcon"] = 0.0
    # Shrinkage target for DefCon rates: the positional median among players
    # with enough minutes to be informative.
    #
    # The threshold must degrade rather than empty out. An earlier version used
    # a hard `minutes > 180`, which selects nobody in the first weeks of a season
    # (and nothing at all when the maximum is exactly 180). The lookup then fell
    # through to a hardcoded 4.0 actions/90 against true medians of ~7.5 for
    # defenders and ~8.4 for midfielders, so every rate was dragged toward a
    # badly wrong prior and threshold probabilities collapsed - silently, since
    # an empty groupby raises nothing.
    dc_k = float(dcfg.get("prior_nineties", 8.0))
    established = df["minutes"] >= 90.0 * dc_k
    if established.sum() < 20:
        established = df["minutes"] >= 90.0
    if established.sum() < 20:
        established = df["minutes"] > 0
    prior_dc = df.loc[established].groupby("position")["dc90"].median().to_dict()
    for pos, grp in df.groupby("position", sort=False):
        pts = float(sc["defensive_contribution"]["pts"].get(pos, 0.0))
        if pts == 0:
            continue
        # Conjugate weight n/(n+k), not a saturating linear rule. The old rule
        # hit full weight at two matches, so an established player received no
        # shrinkage at all - and defensive rates regress to the mean (slope 0.82
        # for defenders), so unshrunk rates over-predicted hits by 9%.
        n90 = np.nan_to_num(grp["minutes"].to_numpy(dtype=float)) / 90.0
        w = n90 / (n90 + dc_k)
        rate = w * np.nan_to_num(grp["dc90"].to_numpy(dtype=float)) + (1.0 - w) * float(
            prior_dc.get(pos, 7.5)
        )
        mix = ev.minutes_mixture(grp["p_60"].values, grp["p_start"].values,
                                 grp["p_play"].values)
        p = ev.defcon_probability(rate, grp["xmins"].values, pos,
                                  grp["opp_lambda"].values, mix=mix)
        df.loc[grp.index, "p_defcon"] = p
        df.loc[grp.index, "xp_defcon"] = p * pts

    # ---- saves -------------------------------------------------------------
    df["exp_saves"] = 0.0
    df["xp_saves"] = 0.0
    gk = df["position"] == "GKP"
    if gk.any():
        gk_mix = ev.minutes_mixture(df.loc[gk, "p_60"].values,
                                    df.loc[gk, "p_start"].values,
                                    df.loc[gk, "p_play"].values)
        df.loc[gk, "xp_saves"] = ev.expected_save_points(
            df.loc[gk, "opp_lambda"].values, df.loc[gk, "xmins"].values, mix=gk_mix
        )
        scfg = params()["saves"]
        df.loc[gk, "exp_saves"] = np.clip(
            float(scfg["intercept"]) + float(scfg["slope"]) * df.loc[gk, "opp_lambda"].values,
            0.0, None,
        ) * np.clip(df.loc[gk, "xmins"].values / 90.0, 0.0, 1.0)

    # ---- cards -------------------------------------------------------------
    # Card rates need shrinking as much as attacking rates do - arguably more,
    # since bookings are rare enough that a single card in a cameo implies an
    # absurd per-90 rate. Shrunk toward the fitted positional average, which is
    # measured over a full season and is a better prior than the current
    # squad's median.
    ccfg = params()["cards"]
    card_k = float(ccfg.get("prior_nineties", 30.0))
    df["xp_cards"] = 0.0
    for pos, grp in df.groupby("position", sort=False):
        base = float(ccfg["yellow_per90"].get(pos, 0.17))
        rate = ev.shrink_rate(
            grp["yc90"].values, grp["minutes"].values, base, 90.0 * card_k
        )
        df.loc[grp.index, "yc90_shrunk"] = rate
        df.loc[grp.index, "xp_cards"] = ev.expected_card_points(
            pos, grp["xmins"].values, rate, grp["closeness"].values
        )

    # ---- penalties ---------------------------------------------------------
    df["xp_pens"] = ev.expected_penalty_points(
        df["is_pen_taker"].fillna(False).values,
        df["team_lambda"].values,
        df["xmins"].values,
    )

    # ---- bonus -------------------------------------------------------------
    df["xp_bonus"] = 0.0
    for pos, grp in df.groupby("position", sort=False):
        df.loc[grp.index, "xp_bonus"] = ev.expected_bonus(
            pos,
            grp["exp_goals"].values,
            grp["exp_assists"].values,
            grp["p_clean_sheet_scored"].values,
            grp["p_defcon"].values,
            grp["exp_saves"].values,
        )

    # ---- total -------------------------------------------------------------
    df["xp"] = df[COMPONENTS].sum(axis=1)
    df.loc[unpriced, ["xp"] + COMPONENTS] = 0.0
    df.loc[unpriced, "xp_source"] = "unpriced"
    return df


def aggregate_to_players(pf: pd.DataFrame) -> pd.DataFrame:
    """Sum (player, fixture) rows to one row per player for the gameweek.

    Doubles add up; blanks never appear and are reinstated at zero by the
    caller if a full player list is needed.
    """
    keys = [
        "player_id", "web_name", "full_name", "team", "team_id", "position",
        "price", "cost_tenths", "status", "news", "selected_by_percent",
        "minutes_source",
    ]
    agg = {c: "sum" for c in COMPONENTS + ["xp", "exp_goals", "exp_assists"]}
    agg.update(
        {
            "fixture_id": "count",
            "opponent": lambda s: "+".join(map(str, s)),
            "is_home": lambda s: "+".join("H" if x else "A" for x in s),
            "xp_source": lambda s: "+".join(sorted(set(map(str, s)))),
            "p_60": "max",
            "p_start": "max",
            "xmins": "max",
        }
    )
    out = pf.groupby(keys, dropna=False).agg(agg).reset_index()
    return out.rename(columns={"fixture_id": "n_fixtures"}).sort_values(
        "xp", ascending=False
    ).reset_index(drop=True)


def provenance_summary(pf: pd.DataFrame) -> pd.DataFrame:
    """How much of the projection rests on exchange data versus a model."""
    rows = []
    total_xp = pf["xp"].sum()
    for src, grp in pf.groupby("xp_source"):
        rows.append(
            {
                "scoreline_source": src,
                "player_fixtures": len(grp),
                "share_of_total_xp": grp["xp"].sum() / total_xp if total_xp else 0.0,
            }
        )
    if "goals_source" in pf.columns:
        for src, grp in pf.groupby("goals_source"):
            rows.append(
                {
                    "scoreline_source": f"[goals] {src}",
                    "player_fixtures": len(grp),
                    "share_of_total_xp": grp["xp_goals"].sum() / max(pf["xp_goals"].sum(), 1e-9),
                }
            )
    return pd.DataFrame(rows)
