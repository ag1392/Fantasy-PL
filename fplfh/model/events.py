"""Per-event expectations, anchored to the exchange wherever possible.

Each function returns an expected count or probability for one scoring event.
Assembly into points happens in ``expected_points.py``.

The organising principle is a tier system, and every number the model produces
carries a tag saying which tier it came from:

  Tier 1  read directly off a liquid exchange market
          (team goals, clean sheets, goals conceded - via the scoreline model)

  Tier 2  read off a thinner exchange market, then *rescaled so it agrees with
          Tier 1*. Anytime-goalscorer books are shallow, but the sum of a
          team's scorer probabilities must reconcile with the team total
          implied by match odds and over/under. Imposing that constraint lets a
          thin market supply relative shape while a liquid one supplies scale.

  Tier 3  no usable market exists, so a statistical model is used with an
          exchange-derived *parameter*. Defensive contribution is the main
          case: no exchange market prices it, but how much defending a side
          does depends on how much it is without the ball, which the match
          odds do tell us.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

from ..config import params

# Fraction of a team's goals that are opposition own goals, and so cannot be
# credited to any FPL player in that team's squad.
OWN_GOAL_FRACTION = 0.025


# --------------------------------------------------------------------------
# Goals
# --------------------------------------------------------------------------
def distribute_team_goals(
    weights: np.ndarray, team_lambda: float, own_goal_fraction: float = OWN_GOAL_FRACTION
) -> np.ndarray:
    """Split a team's expected goals across its players by relative weight.

    The weights are per-player scoring propensity (typically xG per 90 scaled
    by expected minutes). Normalising them to the exchange-implied team total
    is what anchors player-level numbers to liquid market data: individual
    estimates can be noisy, but they are forced to add up to a total the market
    is confident about.
    """
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return np.zeros_like(w)
    return w * (team_lambda * (1.0 - own_goal_fraction) / total)


def goals_from_anytime_market(
    market_probs: np.ndarray,
    team_lambda: float,
    own_goal_fraction: float = OWN_GOAL_FRACTION,
) -> np.ndarray:
    """Convert anytime-goalscorer probabilities to expected goals, rescaled.

    A player's goals in a match are approximately Poisson, so
    ``P(scores at least one) = 1 - exp(-mu)`` inverts to ``mu = -ln(1 - p)``.
    That step matters for the heavy scorers: at p = 0.6, treating the
    probability as the expectation understates them by more than 50%.

    The resulting expectations are then scaled to reconcile with the team total
    from the scoreline model (Tier 2 behaviour).
    """
    p = np.clip(np.asarray(market_probs, dtype=float), 0.0, 0.95)
    mu = -np.log(1.0 - p)
    return distribute_team_goals(mu, team_lambda, own_goal_fraction)


def shrink_rate(
    rate: np.ndarray, minutes: np.ndarray, prior_rate: float, min_minutes: float
) -> np.ndarray:
    """Shrink a per-90 rate toward a prior when the sample behind it is small."""
    rate = np.nan_to_num(np.asarray(rate, dtype=float))
    mins = np.nan_to_num(np.asarray(minutes, dtype=float))
    w = np.clip(mins / max(min_minutes, 1.0), 0.0, 1.0)
    return w * rate + (1.0 - w) * prior_rate


def shrink_rate_grouped(
    df, col: str, prior_nineties: float, group_cols=("position", "price_bucket")
):
    """Shrink a per-90 rate toward its (position, price) peer-group median.

    Per-90 rates computed off a handful of minutes are close to meaningless: a
    single shot taken during a cameo implies a scoring rate no striker sustains.
    Measured on a live squad, xg90 has a standard deviation of ~1.37 among
    players under 90 minutes against ~0.18 for those past 270 - the small-sample
    figure is essentially noise.

    The weight is the Gamma-Poisson conjugate form, ``n / (n + k)``, where n is
    nineties played and k is the prior's strength in the same units. This is the
    right shape for a rate estimate: the information in a Poisson rate scales
    with exposure, so evidence accumulates smoothly and the weight approaches
    but never reaches 1. A linear ``min(minutes / threshold, 1)`` rule instead
    saturates early and hands full credence to three games of data - enough to
    let one penalty make a full-back look like a striker.

    The peer group is position *and* price, because an expensive forward's prior
    genuinely is higher than a cheap defender's, and collapsing both onto a
    single league median would introduce its own bias.
    """
    import numpy as _np

    rate = _np.nan_to_num(df[col].to_numpy(dtype=float))
    nineties = _np.nan_to_num(df["minutes"].to_numpy(dtype=float)) / 90.0

    # Build the prior from players with enough exposure to be informative.
    established = df["minutes"] >= 90.0 * prior_nineties
    source = df.loc[established] if established.sum() >= 20 else df
    prior = source.groupby(list(group_cols), observed=True)[col].median()
    global_prior = float(source[col].median())

    keys = list(zip(*[df[c] for c in group_cols]))
    prior_vals = _np.array(
        [
            float(prior.get(k, _np.nan)) if prior.get(k, _np.nan) == prior.get(k, _np.nan)
            else global_prior
            for k in keys
        ]
    )
    w = nineties / (nineties + max(prior_nineties, 1e-6))
    return w * rate + (1.0 - w) * prior_vals


# --------------------------------------------------------------------------
# Assists
# --------------------------------------------------------------------------
def distribute_team_assists(weights: np.ndarray, team_lambda: float) -> np.ndarray:
    """Split expected team assists across players.

    Total assists are pinned to the exchange-implied team goals via the
    measured league-wide assists-per-goal ratio, so creators inherit the
    market's view of how many goals their side will score.
    """
    ratio = float(params()["assists"]["per_goal"])
    return distribute_team_goals(weights, team_lambda * ratio, own_goal_fraction=0.0)


# --------------------------------------------------------------------------
# Defensive contribution  (Tier 3: model, exchange-parameterised)
# --------------------------------------------------------------------------
# Minutes a player logs in each appearance mode. Shared with the minutes model
# so that `xmins` and the mixture below cannot drift apart.
# Measured over a full season (2025-26): starters reaching 60+ average 85.4
# minutes, starters withdrawn earlier average 47.1, and substitutes 18.2.
# Those three, weighted by how often each happens, reconstruct the 985 minutes
# a team actually uses per match.
APPEARANCE_MODES = {"full": 85.4, "early_sub": 47.1, "cameo": 18.2}


def minutes_mixture(p_60, p_start, p_play):
    """Decompose appearance risk into ``(probability, minutes)`` components.

    Returns the three mutually exclusive ways a player can feature, so that a
    non-linear quantity can be averaged over the minutes *distribution* rather
    than evaluated at expected minutes.
    """
    p_60 = np.clip(np.asarray(p_60, dtype=float), 0.0, 1.0)
    p_start = np.clip(np.asarray(p_start, dtype=float), 0.0, 1.0)
    p_play = np.clip(np.asarray(p_play, dtype=float), 0.0, 1.0)
    return [
        (p_60, APPEARANCE_MODES["full"]),
        (np.clip(p_start - p_60, 0.0, 1.0), APPEARANCE_MODES["early_sub"]),
        (np.clip(p_play - p_start, 0.0, 1.0), APPEARANCE_MODES["cameo"]),
    ]


def on_pitch_clean_sheet(p_cs_90, p_60, minutes_60_plus=None) -> np.ndarray:
    """P(no goal conceded while on the pitch) x P(60+ minutes).

    The scoring rule is not what the name suggests. FPL awards a clean sheet for
    *not conceding while on the pitch*, given 60 minutes played - not for the
    team keeping a clean sheet. A defender substituted on 70 minutes whose side
    concedes on 85 still scores the points.

    Using the team's 90-minute clean sheet probability therefore understates it.
    Measured over 2025-26: teams kept a clean sheet in 25.5% of matches, but
    players who reached 60 minutes were credited with one 28.5% of the time, and
    8.7% of all player clean sheets went to players whose team did not keep one.
    Split by exposure, the mechanism is unambiguous - players going the full 90
    scored at 27.0%, those substituted between 60 and 88 minutes at 32.2%.

    The correction is exact under a Poisson goal process. A side conceding at
    rate lambda over a match concedes at lambda * m / 90 over m minutes, so::

        P(no goal in m minutes) = exp(-lambda * m / 90) = P(clean sheet) ** (m / 90)

    Raising the *fitted* probability to the exposure power rather than rebuilding
    from lambda keeps the Dixon-Coles low-score correction intact, and is exactly
    right at m = 90.

    Averaged over the minutes distribution rather than evaluated at expected
    minutes, for the usual reason: ``p ** (m / 90)`` is convex in m, so the two
    are not the same and the shortcut is biased low. Only the 60-plus branch can
    score, so the mixture reduces to that single mode.
    """
    p_cs = np.clip(np.asarray(p_cs_90, dtype=float), 1e-9, 1.0)
    m = APPEARANCE_MODES["full"] if minutes_60_plus is None else minutes_60_plus
    return np.asarray(p_60, dtype=float) * p_cs ** (np.asarray(m, dtype=float) / 90.0)


def defcon_probability(
    dc_rate90: np.ndarray,
    xmins: np.ndarray,
    position: str,
    opponent_lambda: np.ndarray | float,
    mix=None,
) -> np.ndarray:
    """P(player reaches the defensive-contribution threshold).

    No exchange market prices defensive actions, so this is a statistical model
    - but the exchange still supplies a parameter. A side expected to concede
    more is a side that will spend more time defending, and its players
    accumulate tackles, interceptions and recoveries faster. The rate is
    therefore scaled by the opponent's expected goals relative to the league
    average, raised to a damping exponent.

    Counts are negative binomial. The dispersion was solved, per position, so
    that the modelled threshold hit-rate reproduces the rate actually observed
    over a full season. A plain Poisson understates it by about 1.4 points of
    probability, which biases every defender's projection the same way.

    **Minutes are averaged over, not plugged in.** Pass ``mix`` (from
    ``minutes_mixture``) and the threshold probability is evaluated separately
    for each appearance mode and weighted, which is ``E[f(minutes)]``. Passing
    only ``xmins`` computes ``f(E[minutes])`` instead - and the two differ a
    great deal here, because P(X >= threshold) is strongly convex in minutes.
    A player with a 40% chance of 85 minutes is not equivalent to one certain to
    play 34: the first can reach the threshold, the second essentially cannot.
    Backtesting showed the point estimate producing 21 expected league-wide hits
    against 48 on the same rates with real minutes.
    """
    cfg = params()["defcon"]
    thresholds = params()["scoring"]["defensive_contribution"]["thresholds"]
    if position not in thresholds:
        return np.zeros_like(np.asarray(dc_rate90, dtype=float))

    thr = int(thresholds[position])
    phi = float(cfg["dispersion"].get(position, 1.3))
    beta = float(cfg["opponent_scaling_beta"])
    avg_lambda = float(cfg["league_avg_lambda"])

    opp = np.clip(np.asarray(opponent_lambda, dtype=float), 0.2, 5.0)
    scale = (opp / avg_lambda) ** beta
    rate = np.clip(np.asarray(dc_rate90, dtype=float), 0.0, None) * scale

    def _tail(minutes) -> np.ndarray:
        mu = np.clip(rate * (np.asarray(minutes, dtype=float) / 90.0), 1e-6, None)
        if phi <= 1.0 + 1e-9:
            return poisson.sf(thr - 1, mu)
        r = mu / (phi - 1.0)
        return nbinom.sf(thr - 1, r, r / (r + mu))

    if mix is None:
        return _tail(xmins)
    return sum(np.asarray(w, dtype=float) * _tail(m) for w, m in mix)


# --------------------------------------------------------------------------
# Saves  (Tier 3: model, exchange-parameterised)
# --------------------------------------------------------------------------
def expected_save_points(
    opponent_lambda: np.ndarray, xmins: np.ndarray, mix=None
) -> np.ndarray:
    """E[floor(saves / 3)] for a goalkeeper.

    Save volume is driven by how much shooting the opponent does, which the
    exchange prices directly as the opponent's expected goals. The fitted
    relationship is affine, and the intercept carries real weight: a keeper
    makes roughly 1.4 saves even in a quiet game, so a through-the-origin fit
    would badly underrate keepers in low-scoring fixtures.

    The floor is evaluated exactly over the count distribution rather than
    applied to the mean, because floor() is concave and E[floor(S/3)] sits
    meaningfully below floor(E[S]/3).
    """
    cfg = params()["saves"]
    per_n = int(params()["scoring"]["saves"]["per_n_saves"])
    pts = float(params()["scoring"]["saves"]["pts"])

    lam = np.clip(np.asarray(opponent_lambda, dtype=float), 0.0, 6.0)
    per_90 = np.clip(float(cfg["intercept"]) + float(cfg["slope"]) * lam, 0.0, None)

    k = np.arange(0, 25)
    floor_k = np.floor(k / per_n)

    def _pts(minutes) -> np.ndarray:
        frac = np.clip(np.asarray(minutes, dtype=float) / 90.0, 0.0, 1.0)
        mean = np.atleast_1d(per_90 * frac)
        pmf = poisson.pmf(k[None, :], mean[:, None])
        return (pmf * floor_k[None, :]).sum(axis=1) * pts

    # As with defensive contribution, the floor makes this non-linear in minutes,
    # so the minutes distribution is averaged over rather than collapsed first.
    if mix is None:
        return _pts(xmins)
    return sum(np.asarray(w, dtype=float) * _pts(m) for w, m in mix)


# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------
def expected_card_points(
    position: str, xmins: np.ndarray, yc_rate90: np.ndarray | None = None,
    closeness: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Expected points from yellow and red cards (a negative number).

    Tight matches produce more cards, so the rate is nudged by how close the
    exchange thinks the game is. The effect is small - typically -0.1 to -0.2
    points - but it is systematic, and it falls hardest on exactly the
    aggressive defenders and holding midfielders a defensive-contribution model
    otherwise rates highly.
    """
    cfg = params()["cards"]
    sc = params()["scoring"]
    mins_frac = np.clip(np.asarray(xmins, dtype=float) / 90.0, 0.0, 1.0)

    base_yc = float(cfg["yellow_per90"].get(position, 0.17))
    rate = base_yc if yc_rate90 is None else np.asarray(yc_rate90, dtype=float)
    rate = np.nan_to_num(rate, nan=base_yc)

    adj = np.asarray(closeness, dtype=float) ** float(cfg["closeness_beta"])
    mu_yc = np.clip(rate * mins_frac * adj, 0.0, 2.0)
    mu_rc = float(cfg["red_per90"].get(position, 0.005)) * mins_frac

    p_yc = 1.0 - np.exp(-mu_yc)
    p_rc = 1.0 - np.exp(-mu_rc)
    return p_yc * float(sc["yellow_cards"]) + p_rc * float(sc["red_cards"])


def match_closeness(p_home: float, p_draw: float, p_away: float) -> float:
    """A scalar in roughly [0.5, 1.5]: higher means a tighter, more fractious game.

    Defined from the entropy of the result distribution, normalised so an
    evenly-poised match sits at 1.0.
    """
    ps = np.clip(np.array([p_home, p_draw, p_away], dtype=float), 1e-9, 1.0)
    ps = ps / ps.sum()
    entropy = -(ps * np.log(ps)).sum()
    return float(entropy / np.log(3.0))


# --------------------------------------------------------------------------
# Bonus
# --------------------------------------------------------------------------
def expected_bonus(
    position: str,
    exp_goals: np.ndarray,
    exp_assists: np.ndarray,
    p_clean_sheet: np.ndarray,
    p_defcon: np.ndarray,
    exp_saves: np.ndarray,
) -> np.ndarray:
    """Expected bonus points.

    Bonus is awarded on the BPS ranking within a match, which depends on data
    (passes, dribbles, shots blocked) this model does not project. Rather than
    reconstruct BPS, the relationship is fitted directly onto the events the
    model *does* project. Because the fit is linear, it can be applied to
    expectations rather than realisations without introducing bias.
    """
    coeffs = params()["bonus"]["coefficients"].get(position)
    if not coeffs:
        return np.zeros_like(np.asarray(exp_goals, dtype=float))
    out = (
        float(coeffs["goals"]) * np.asarray(exp_goals, dtype=float)
        + float(coeffs["assists"]) * np.asarray(exp_assists, dtype=float)
        + float(coeffs["clean_sheet"]) * np.asarray(p_clean_sheet, dtype=float)
        + float(coeffs["defcon_hit"]) * np.asarray(p_defcon, dtype=float)
        + float(coeffs["saves"]) * np.asarray(exp_saves, dtype=float)
        + float(coeffs["const"])
    )
    return np.clip(out, 0.0, 3.0)


# --------------------------------------------------------------------------
# Penalties
# --------------------------------------------------------------------------
def expected_penalty_points(
    is_pen_taker: np.ndarray, team_lambda: np.ndarray, xmins: np.ndarray | None = None
) -> np.ndarray:
    """Points lost to missed penalties by the designated taker.

    Penalty *goals* are already inside the goal expectation (Opta xG includes
    them, and anytime-scorer prices price them in), so only the miss deduction
    is added here - double-counting the upside would inflate every penalty
    taker.

    The deduction is scaled by minutes, because a taker can only take a penalty
    while on the pitch. Omitting that scaling hands a deduction to injured or
    suspended takers for matches they will not appear in, leaving them with a
    *negative* projected total made up entirely of a penalty they can never
    miss.
    """
    sc = params()["scoring"]
    taker = np.asarray(is_pen_taker, dtype=float)
    # League-wide: roughly 0.13 penalties awarded per team-match, converting at
    # about 79%. Scaled by attacking volume, since better sides win more.
    pens_per_match = 0.13 * np.clip(np.asarray(team_lambda, dtype=float) / 1.45, 0.4, 2.0)
    share = (
        1.0 if xmins is None
        else np.clip(np.asarray(xmins, dtype=float) / 90.0, 0.0, 1.0)
    )
    p_miss = pens_per_match * (1.0 - 0.79) * taker * share
    return p_miss * float(sc["penalties_missed"])
