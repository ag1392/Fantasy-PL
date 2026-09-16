"""Expected minutes.

Minutes are the single largest source of variance in an FPL projection, and
the one place where a human with team-news knowledge reliably beats a model.
So this module does two things:

1. Builds an *automatic prior* from data the FPL API already gives away free -
   availability status, chance-of-playing percentage, and the recent
   start/minutes record - with empirical-Bayes shrinkage so that a player with
   two appearances is not treated as confidently as one with twenty.

2. Lets a human override any of it, per player, in
   ``config/minutes_overrides.yaml``.

The shrinkage target is learned from the data itself rather than hardcoded:
players are bucketed by price, and each bucket's observed start rate becomes
the prior for players in it. Expensive players do start more often, and this
picks that up without anyone having to assert a number.

Everything downstream consumes three quantities:
    p_play  - P(appears at all)
    p_60    - P(reaches 60 minutes)      -> the 2-point appearance band, and
                                            the precondition for a clean sheet
    xmins   - expected minutes           -> scales all per-90 event rates
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import overrides, params


def build_minutes_history(client, events: list[int] | None = None,
                          max_age: float | None = None) -> pd.DataFrame:
    """Per-player, per-gameweek minutes from the cached ``live`` endpoints.

    Uses the gameweek live feeds rather than 658 individual element-summary
    calls - same information, two orders of magnitude fewer requests.

    ``max_age`` is passed straight through to each ``client.live()`` call, so
    a caller forcing a full refresh (``max_age=0``) also re-fetches these -
    otherwise they sit at the client's own default (one hour).
    """
    events = events or client.finished_events()
    rows = []
    for ev in events:
        for el in client.live(ev, max_age=max_age)["elements"]:
            st = el["stats"]
            rows.append(
                {
                    "player_id": el["id"],
                    "event": ev,
                    "minutes": st["minutes"],
                    "started": int(st.get("starts", 0) or 0),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["player_id", "event", "minutes", "started"])
    return pd.DataFrame(rows)


def _availability(row) -> float:
    """Multiplier in [0, 1] from the FPL availability flags.

    ``chance_of_playing`` is authoritative when present (FPL sets it from press
    conferences). Otherwise fall back to the status code.
    """
    cfg = params()["minutes"]["status_availability"]
    chance = row.get("chance_of_playing")
    if chance is not None and not (isinstance(chance, float) and np.isnan(chance)):
        return float(chance) / 100.0
    return float(cfg.get(row.get("status", "a"), 1.0))


def estimate_minutes(
    players: pd.DataFrame,
    history: pd.DataFrame,
    target_event: int,
    apply_overrides: bool = True,
) -> pd.DataFrame:
    """Return players with ``p_start``, ``p_play``, ``p_60``, ``xmins`` columns.

    Manual overrides from ``config/minutes_overrides.yaml`` are applied last, so
    they always win. To pin a player to a full match, set ``minutes: 90`` there.
    """
    cfg = params()["minutes"]
    lookback = int(cfg["lookback_gws"])
    halflife = float(cfg["recency_halflife_gws"])
    pseudo = float(cfg["prior_pseudocount"])
    p60_start_default = float(cfg["p60_given_start_default"])

    out = players.copy()

    # ---------------------------------------------------------- recent form
    recent = history[
        (history["event"] < target_event) & (history["event"] >= target_event - lookback)
    ].copy()

    if recent.empty:
        out["obs_n"] = 0.0
        out["obs_start_rate"] = np.nan
        out["obs_p60_given_start"] = np.nan
    else:
        # Exponential recency weights: a game one half-life ago counts half as
        # much as the most recent one.
        age = target_event - recent["event"]
        recent["w"] = 0.5 ** (age / halflife)
        # Rescale so the weights sum to the actual number of appearances. Without
        # this the weighting quietly discards information - three observed games
        # would carry an effective sample size of only ~1.5, letting the prior
        # swamp genuine signal and collapsing distinct players onto one value.
        scale = recent.groupby("player_id")["w"].transform(lambda s: len(s) / s.sum())
        recent["w"] = recent["w"] * scale
        recent["w_start"] = recent["w"] * recent["started"]
        recent["w_60"] = recent["w"] * ((recent["minutes"] >= 60) & (recent["started"] == 1))

        g = recent.groupby("player_id")
        agg = pd.DataFrame(
            {
                "obs_n": g["w"].sum(),
                "w_starts": g["w_start"].sum(),
                "w_60_starts": g["w_60"].sum(),
            }
        )
        agg["obs_start_rate"] = agg["w_starts"] / agg["obs_n"].where(agg["obs_n"] > 0)
        agg["obs_p60_given_start"] = np.where(
            agg["w_starts"] > 0, agg["w_60_starts"] / agg["w_starts"].where(agg["w_starts"] > 0), np.nan
        )
        out = out.merge(
            agg[["obs_n", "obs_start_rate", "obs_p60_given_start"]],
            left_on="player_id",
            right_index=True,
            how="left",
        )
        out["obs_n"] = out["obs_n"].fillna(0.0)

    # ------------------------------------------- empirical-Bayes price prior
    # Learn the shrinkage target from the data instead of asserting it.
    out["price_bucket"] = pd.cut(
        out["price"],
        bins=[0, 4.4, 4.9, 5.4, 6.4, 7.9, 30.0],
        labels=["<=4.4", "4.5-4.9", "5.0-5.4", "5.5-6.4", "6.5-7.9", "8.0+"],
    )
    have = out["obs_n"] > 0
    grp = out.loc[have].groupby(["position", "price_bucket"], observed=True)["obs_start_rate"]
    bucket_mean, bucket_n = grp.mean(), grp.size()
    global_prior = float(out.loc[have, "obs_start_rate"].mean()) if have.any() else 0.35

    # Second level of shrinkage: pull each bucket's mean toward the global mean.
    # Without this, a small homogeneous bucket (say, the four forwards priced
    # 8.0m+, all of whom have started every game) yields a prior of exactly 1.0,
    # which then propagates into p_start = 1.0 - asserting a player is certain
    # to start, which is never true.
    bucket_k = float(cfg.get("bucket_shrinkage_k", 10.0))
    prior_map = (bucket_mean * bucket_n + global_prior * bucket_k) / (bucket_n + bucket_k)

    def _prior(row) -> float:
        val = prior_map.get((row["position"], row["price_bucket"]), np.nan)
        return float(val) if val == val else global_prior

    out["start_prior"] = out.apply(_prior, axis=1)

    # ------------------------------------------------------------ shrinkage
    n = out["obs_n"].fillna(0.0)
    obs = out["obs_start_rate"].fillna(out["start_prior"])
    out["p_start_raw"] = (obs * n + out["start_prior"] * pseudo) / (n + pseudo)

    p60_obs = out["obs_p60_given_start"].fillna(p60_start_default)
    starts_n = (n * out["p_start_raw"]).clip(lower=0)
    out["p60_given_start"] = (p60_obs * starts_n + p60_start_default * pseudo) / (
        starts_n + pseudo
    )

    # ---------------------------------------------------------- availability
    out["availability"] = out.apply(_availability, axis=1)

    # Ceiling on certainty: even a nailed-on starter carries irreducible risk of
    # a late knock, an illness or a surprise rotation. Letting p_start reach 1.0
    # would price that risk at exactly zero.
    ceiling = float(cfg.get("p_start_ceiling", 0.97))
    out["p_start"] = (out["p_start_raw"] * out["availability"]).clip(0.0, ceiling)


    # A team starts exactly eleven players, and nothing above enforces that.
    # Shrinkage moves each player's estimate on its own merits, so the team
    # total drifts - measured at 9.85 before this was added, which understated
    # every clean sheet and appearance point by the same proportion.
    if cfg.get("normalise_team_starts", True):
        out = _normalise_team_starts(out, target=11.0, ceiling=ceiling)
    # A non-starter who is in the squad may still appear off the bench, with a
    # probability that rises with how close he is to the team - see
    # bench_probability. The team total is then pinned the same way starts are.
    bench = pd.Series(
        bench_probability(out["p_start"], out["availability"]), index=out.index
    ) * (1.0 - out["p_start"])
    out["p_play"] = _normalise_team_subs(
        out, (out["p_start"] + bench).clip(0.0, 1.0)
    )
    out["p_60"] = (out["p_start"] * out["p60_given_start"]).clip(0.0, 1.0)

    # Expected minutes, using the same measured constants as the event models
    # so the two cannot drift apart.
    from .events import APPEARANCE_MODES as _AM

    p_cameo = (out["p_play"] - out["p_start"]).clip(lower=0.0)
    p_start_short = (out["p_start"] - out["p_60"]).clip(lower=0.0)
    out["xmins"] = (out["p_60"] * _AM["full"] + p_start_short * _AM["early_sub"]
                    + p_cameo * _AM["cameo"])

    out["minutes_source"] = "auto"

    # -------------------------------------------------------- human override
    if apply_overrides:
        out = _apply_overrides(out)

    out["p_60"] = np.minimum(out["p_60"], out["p_play"])
    return out


def bench_probability(p_start, availability) -> np.ndarray:
    """P(appears off the bench | did not start), as a function of standing.

    The flat constant this replaced said a player who starts 3% of the time
    still appears 37% of the time. He appears 7% of the time. The error was
    invisible in aggregate because the constant had been calibrated so that
    *team totals* came out right on a pool already thinned by availability
    flags - so it was simultaneously far too high for fringe players and too
    low for rotation players, and the two cancelled.

    The shape is not assumed. Fitted by maximum likelihood over 21,395
    non-start observations in 2025-26, P(sub | did not start) rises steeply
    from near zero and saturates::

        p = floor + (p_max - floor) * s / (s + half_at)

    where ``s`` is the player's start probability. Fit against observation:

        start rate   n        observed   fitted
        0.00-0.05    13,259   0.028      0.026
        0.05-0.15     1,758   0.205      0.228
        0.15-0.30     2,268   0.336      0.335
        0.30-0.50     2,011   0.367      0.382
        0.50-0.70     1,421   0.448      0.407
        0.70-0.90       611   0.403      0.423

    The intuition it encodes: a player the manager is willing to start is also
    the player he brings on, while a player who never starts is usually not in
    the matchday squad at all. Modelling the bench as *leftover* probability -
    which is what ``(availability - p_start) * constant`` does - has it exactly
    backwards, handing the most bench weight to the player furthest from the team.

    One known limitation, left in place: at start rates above 0.9 the observed
    rate falls back to 0.24, because a nailed starter who does not start is
    usually injured or rested rather than benched. That band holds 67
    observations, and the curve holds them at 0.43. The cost is small - those
    players carry p_start near the ceiling, so the non-start branch barely
    contributes - but it is an overstatement, not noise.
    """
    cfg = params()["minutes"]["sub_curve"]
    floor, p_max = float(cfg["floor"]), float(cfg["p_max"])
    half_at = float(cfg["half_at"])
    s = np.asarray(p_start, dtype=float)
    p = floor + (p_max - floor) * s / (s + half_at)
    # An unavailable player cannot be on the bench either.
    return np.clip(p * np.asarray(availability, dtype=float), 0.0, 1.0)


def _normalise_team_subs(df: pd.DataFrame, p_play: pd.Series) -> pd.Series:
    """Scale bench probabilities so each team's sub appearances sum to target.

    The same argument as ``_normalise_team_starts``, applied to the other half
    of the appearance model. A match allows five substitutions and 4.19 distinct
    substitutes appeared per team-match in 2025-26; the per-player curve cannot
    see that constraint, so it is imposed here.

    Unlike starts, this is a constraint on an *expectation*, not a certainty:
    the number of substitutes used varies (sd 1.14). Pinning the mean is still
    right, and it keeps the total honest when availability flags thin the pool.
    """
    cfg = params()["minutes"]
    target = float(cfg.get("subs_per_team", 4.19))
    if not cfg.get("normalise_team_subs", True):
        return p_play

    out = p_play.copy()
    bench = (p_play - df["p_start"]).clip(lower=0.0)
    for _, grp in df.groupby("team_id", sort=False):
        idx = grp.index
        b = bench.loc[idx].to_numpy(dtype=float, copy=True)
        total = b.sum()
        if total <= 0:
            continue
        headroom = (1.0 - df.loc[idx, "p_start"].to_numpy(float)).clip(0.0, 1.0)
        scale = target / total
        b = np.minimum(b * scale, headroom)
        out.loc[idx] = df.loc[idx, "p_start"].to_numpy(float) + b
    return out.clip(0.0, 1.0)


def _normalise_team_starts(
    df: pd.DataFrame, target: float = 11.0, ceiling: float = 0.97, iters: int = 50
) -> pd.DataFrame:
    """Force each team's start probabilities to sum to eleven.

    This is a hard constraint that the per-player model cannot see. Every
    estimate is individually sensible, but shrinkage pulls nailed starters down
    and fringe players up, and nothing makes the squad add up to a team.

    Scaling is proportional and respects the certainty ceiling, iterating so
    that probability displaced by a capped player is redistributed among the
    others rather than lost. Players at zero - injured, suspended - stay at
    zero, since multiplying by zero is still zero; their share correctly passes
    to whoever plays instead.
    """
    for _, grp in df.groupby("team_id", sort=False):
        idx = grp.index
        p = df.loc[idx, "p_start"].to_numpy(dtype=float, copy=True)
        for _ in range(iters):
            total = p.sum()
            if total <= 0 or abs(total - target) < 1e-9:
                break
            free = p < ceiling - 1e-12
            base = p[free].sum()
            if not free.any() or base <= 0:
                break
            p[free] = np.clip(p[free] * (1.0 + (target - total) / base), 0.0, ceiling)
        df.loc[idx, "p_start"] = p
    return df


def _apply_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Apply config/minutes_overrides.yaml.

    Keys may be a player id, a web_name, or "web_name (TEAM)" to disambiguate.
    Each entry may set any of: p_start, p_play, p_60, xmins, or the shorthand
    ``minutes`` (expected minutes, from which the probabilities are implied).
    """
    ov = overrides()
    entries = ov.get("players") or {}
    if not entries:
        return df

    by_id = {str(r.player_id): i for i, r in df.iterrows()}
    by_name: dict[str, list[int]] = {}
    for i, r in df.iterrows():
        by_name.setdefault(str(r.web_name).lower(), []).append(i)
        by_name.setdefault(f"{str(r.web_name).lower()} ({str(r.team).lower()})", []).append(i)

    unmatched: list[str] = []
    for key, spec in entries.items():
        k = str(key).strip()
        idxs: list[int] = []
        if k in by_id:
            idxs = [by_id[k]]
        elif k.lower() in by_name:
            idxs = by_name[k.lower()]
        if not idxs:
            unmatched.append(k)
            continue
        if len(idxs) > 1:
            unmatched.append(f"{k} (ambiguous: {len(idxs)} players - qualify with team)")
            continue
        i = idxs[0]
        if spec is None:
            continue
        if not isinstance(spec, dict):
            # Shorthand: "Salah: 0" means expected minutes 0 (ruled out).
            spec = {"minutes": float(spec)}
        if "minutes" in spec:
            m = float(spec["minutes"])
            df.at[i, "xmins"] = m
            df.at[i, "p_play"] = 1.0 if m > 0 else 0.0
            df.at[i, "p_start"] = 1.0 if m >= 60 else (0.0 if m == 0 else 0.5)
            df.at[i, "p_60"] = 1.0 if m >= 60 else 0.0
        for col in ("p_start", "p_play", "p_60", "xmins"):
            if col in spec:
                df.at[i, col] = float(spec[col])
        df.at[i, "minutes_source"] = "override"

    if unmatched:
        df.attrs["override_warnings"] = unmatched
    return df


def minutes_points(df: pd.DataFrame) -> pd.Series:
    """Expected appearance points.

    1 point for 1-59 minutes, 2 for 60+, so the expectation collapses neatly to
    ``p_play + p_60``.
    """
    s = params()["scoring"]["minutes"]
    return df["p_play"] * float(s["short_play_pts"]) + df["p_60"] * (
        float(s["long_play_pts"]) - float(s["short_play_pts"])
    )


def minutes_for_event(
    base: pd.DataFrame,
    event: int,
    return_gw: "pd.Series | None" = None,
) -> pd.DataFrame:
    """Re-derive minutes for a *future* gameweek from an already-built frame.

    Multi-gameweek projections need availability to change over the horizon: a
    player flagged today may be back by the gameweek in question. Where FPL
    states a return date, ``return_gw`` maps player id to the first gameweek
    they could feature in, and from that point their availability is restored.

    Everyone else is frozen at today's availability - FPL gives no return date
    for most flagged players, and inventing a recovery curve would present a
    guess as information.

    Manual overrides are never touched: if you have pinned a player's minutes,
    that stands for every gameweek in the window.
    """
    cfg = params()["minutes"]
    ceiling = float(cfg.get("p_start_ceiling", 0.97))

    out = base.copy()
    out["availability_event"] = out["availability"]

    out["minutes_basis"] = "observed"
    base_rate = out["p_start_raw"].copy()

    if return_gw is not None and len(return_gw):
        back = out["player_id"].map(return_gw)
        # Restored players return to their baseline start rate, not to certainty
        # - being fit again is not the same as being picked.
        restored = back.notna() & (event >= back.fillna(10**6))
        out.loc[restored, "availability_event"] = 1.0

        # Their observed record cannot be used as that baseline. A player who
        # has been injured has a start rate of zero *because* of the injury -
        # the zeros are not missing at random, and they say nothing about
        # whether he would be picked when fit. Several of these have played no
        # minutes at all this season, so there is no uncontaminated in-season
        # evidence to fall back on either.
        #
        # What is left is the (position, price) prior, which at least encodes
        # that an expensive player is usually a starter. It is crude, so these
        # rows are tagged and the notebook flags them.
        base_rate = base_rate.where(~restored, out["start_prior"])
        out.loc[restored, "minutes_basis"] = "prior (returning from injury)"
        out.attrs["restored_count"] = int(restored.sum())

    overridden = out.get("minutes_source", pd.Series("auto", index=out.index)) == "override"

    p_start = (base_rate * out["availability_event"]).clip(0.0, ceiling)
    out.loc[~overridden, "p_start"] = p_start[~overridden]

    if cfg.get("normalise_team_starts", True):
        out = _normalise_team_starts(out, target=11.0, ceiling=ceiling)

    from .events import APPEARANCE_MODES as _AM

    p60 = (out["p_start"] * out["p60_given_start"]).clip(0.0, 1.0)
    bench = pd.Series(
        bench_probability(out["p_start"], out["availability_event"]), index=out.index
    ) * (1.0 - out["p_start"])
    p_play = _normalise_team_subs(out, (out["p_start"] + bench).clip(0.0, 1.0))
    cameo = (p_play - out["p_start"]).clip(lower=0.0)
    short = (out["p_start"] - p60).clip(lower=0.0)
    xmins = p60 * _AM["full"] + short * _AM["early_sub"] + cameo * _AM["cameo"]

    for col, vals in (("p_60", p60), ("p_play", p_play), ("xmins", xmins)):
        out.loc[~overridden, col] = vals[~overridden]
    out["p_60"] = np.minimum(out["p_60"], out["p_play"])
    return out
