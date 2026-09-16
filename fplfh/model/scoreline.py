"""Match scoreline distribution - the bridge from exchange prices to FPL points.

This is the core of the exchange-anchored approach. Rather than reading each
FPL-relevant quantity off its own (usually illiquid) market, we fit a single
Dixon-Coles bivariate goal model to the *most liquid* markets on the exchange -
match odds, over/under lines, both-teams-to-score - and then read every
team-level FPL quantity off the resulting joint distribution:

    clean sheet probability      P(opponent goals = 0)
    goals-conceded penalty       E[floor(goals against / 2)]
    team attacking volume        lambda, used to anchor player goal rates
    save volume                  a function of opponent lambda

That is a much better trade than pricing each quantity separately: the markets
we lean on have millions matched, the model has three parameters, and every
derived quantity is automatically mutually consistent (clean sheets, match
result and total goals cannot disagree with one another, which they can when
each is read off a separate book).

Dixon-Coles adds a low-score correction to independent Poisson, because real
football has more 0-0 and 1-1 draws than independence implies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

from ..config import params


# --------------------------------------------------------------------------
# Joint distribution
# --------------------------------------------------------------------------
@dataclass
class ScorelineDistribution:
    """Joint distribution over (home goals, away goals)."""

    lambda_home: float
    lambda_away: float
    rho: float
    joint: np.ndarray                 # joint[h, a] = P(home=h, away=a)
    source: str                       # provenance, e.g. "oddsapi:betfair_ex_uk:h2h+totals"
    fit_residual: float = 0.0
    markets_used: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    # ---------------------------------------------------------- marginals
    def goals_for(self, is_home: bool) -> np.ndarray:
        """Marginal distribution of goals scored by one side."""
        return self.joint.sum(axis=1 if is_home else 0)

    def goals_against(self, is_home: bool) -> np.ndarray:
        """Marginal distribution of goals conceded by one side."""
        return self.goals_for(not is_home)

    def expected_goals(self, is_home: bool) -> float:
        m = self.goals_for(is_home)
        return float(np.dot(np.arange(len(m)), m))

    def expected_conceded(self, is_home: bool) -> float:
        return self.expected_goals(not is_home)

    # ------------------------------------------------------- FPL quantities
    def p_clean_sheet(self, is_home: bool) -> float:
        """P(this side concedes zero)."""
        return float(self.goals_against(is_home)[0])

    def expected_conceded_penalty_units(self, is_home: bool, per_n: int = 2) -> float:
        """E[floor(goals conceded / per_n)].

        FPL deducts a point per *two* goals conceded, so the expectation of the
        floor is required - using E[conceded]/2 instead would overstate the
        deduction, because floor() is concave here.
        """
        against = self.goals_against(is_home)
        k = np.arange(len(against))
        return float(np.dot(np.floor(k / per_n), against))

    def p_win(self, is_home: bool) -> float:
        idx = np.arange(self.joint.shape[0])
        mask = idx[:, None] > idx[None, :]
        return float(self.joint[mask].sum() if is_home else self.joint[mask.T].sum())

    def p_draw(self) -> float:
        return float(np.trace(self.joint))

    def p_over(self, line: float) -> float:
        idx = np.arange(self.joint.shape[0])
        total = idx[:, None] + idx[None, :]
        return float(self.joint[total > line].sum())

    def p_btts(self) -> float:
        return float(self.joint[1:, 1:].sum())

    def summary(self) -> dict:
        return {
            "lambda_home": round(self.lambda_home, 3),
            "lambda_away": round(self.lambda_away, 3),
            "rho": round(self.rho, 4),
            "p_home": round(self.p_win(True), 4),
            "p_draw": round(self.p_draw(), 4),
            "p_away": round(self.p_win(False), 4),
            "p_over_2.5": round(self.p_over(2.5), 4),
            "p_btts": round(self.p_btts(), 4),
            "cs_home": round(self.p_clean_sheet(True), 4),
            "cs_away": round(self.p_clean_sheet(False), 4),
            "source": self.source,
            "fit_residual": round(self.fit_residual, 5),
        }


# --------------------------------------------------------------------------
# Dixon-Coles machinery
# --------------------------------------------------------------------------
def _tau(h: np.ndarray, a: np.ndarray, lh: float, la: float, rho: float) -> np.ndarray:
    """Low-score dependency correction (Dixon & Coles 1997)."""
    t = np.ones_like(h, dtype=float)
    t = np.where((h == 0) & (a == 0), 1.0 - lh * la * rho, t)
    t = np.where((h == 0) & (a == 1), 1.0 + lh * rho, t)
    t = np.where((h == 1) & (a == 0), 1.0 + la * rho, t)
    t = np.where((h == 1) & (a == 1), 1.0 - rho, t)
    return t


def build_joint(lh: float, la: float, rho: float, max_goals: int | None = None) -> np.ndarray:
    """Joint P(h, a) matrix, renormalised to sum to 1."""
    max_goals = max_goals or int(params()["scoreline"]["max_goals"])
    n = max_goals + 1
    ph = poisson.pmf(np.arange(n), max(lh, 1e-6))
    pa = poisson.pmf(np.arange(n), max(la, 1e-6))
    joint = np.outer(ph, pa)
    hh, aa = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    joint = joint * _tau(hh, aa, lh, la, rho)
    joint = np.clip(joint, 0.0, None)
    total = joint.sum()
    return joint / total if total > 0 else joint


# --------------------------------------------------------------------------
# Fit to exchange markets
# --------------------------------------------------------------------------
def fit_to_markets(
    p_home: float | None = None,
    p_draw: float | None = None,
    p_away: float | None = None,
    over_lines: dict[float, float] | None = None,
    p_btts: float | None = None,
    source: str = "market",
    markets_used: list[str] | None = None,
) -> ScorelineDistribution:
    """Fit (lambda_home, lambda_away, rho) to de-vigged exchange probabilities.

    Every argument is optional; supply whatever markets were liquid. Two
    observations are enough to identify the two lambdas, and anything beyond
    that constrains rho and over-determines the fit (which is a good thing -
    the residual becomes a usable diagnostic of market disagreement).

    ``over_lines`` maps a goal line to P(total goals > line), e.g.
    ``{2.5: 0.54}``.
    """
    over_lines = over_lines or {}
    targets: list[tuple[str, float, callable]] = []

    if p_home is not None:
        targets.append(("home", p_home, lambda d: d.p_win(True)))
    if p_draw is not None:
        targets.append(("draw", p_draw, lambda d: d.p_draw()))
    if p_away is not None:
        targets.append(("away", p_away, lambda d: d.p_win(False)))
    for line, prob in sorted(over_lines.items()):
        targets.append((f"over_{line}", prob, (lambda ln: (lambda d: d.p_over(ln)))(line)))
    if p_btts is not None:
        targets.append(("btts", p_btts, lambda d: d.p_btts()))

    if len(targets) < 2:
        raise ValueError(
            "need at least two market observations to fit a scoreline model; "
            f"got {len(targets)}"
        )

    max_goals = int(params()["scoreline"]["max_goals"])

    def residuals(theta: np.ndarray) -> np.ndarray:
        lh, la, rho = math.exp(theta[0]), math.exp(theta[1]), theta[2]
        joint = build_joint(lh, la, rho, max_goals)
        d = ScorelineDistribution(lh, la, rho, joint, source)
        return np.array([fn(d) - obs for _, obs, fn in targets])

    # Sensible start: ~1.5 goals each, mild positive low-score dependency.
    x0 = np.array([math.log(1.5), math.log(1.2), -0.03])
    sol = least_squares(
        residuals,
        x0,
        bounds=([math.log(0.05), math.log(0.05), -0.25], [math.log(6.0), math.log(6.0), 0.25]),
        xtol=1e-12,
        ftol=1e-12,
    )
    lh, la, rho = math.exp(sol.x[0]), math.exp(sol.x[1]), float(sol.x[2])
    joint = build_joint(lh, la, rho, max_goals)
    resid = float(np.sqrt(np.mean(sol.fun**2)))

    dist = ScorelineDistribution(
        lambda_home=lh,
        lambda_away=la,
        rho=rho,
        joint=joint,
        source=source,
        fit_residual=resid,
        markets_used=markets_used or [],
    )
    dist.diagnostics = {
        name: {"market": round(obs, 4), "model": round(fn(dist), 4)}
        for name, obs, fn in targets
    }
    return dist


def from_correct_score(
    cs_probs: dict[tuple[int, int], float],
    residual_home: float = 0.0,
    residual_away: float = 0.0,
    residual_draw: float = 0.0,
    source: str = "market:CORRECT_SCORE",
) -> ScorelineDistribution:
    """Build a distribution straight from a de-vigged correct-score book.

    Correct-score markets typically enumerate scorelines up to 3-3 and bucket the
    rest into "any other home win / draw / away win". Those buckets are spread
    over the unenumerated cells in proportion to an independent-Poisson shape
    fitted to the enumerated part, which keeps the totals exact.
    """
    max_goals = int(params()["scoreline"]["max_goals"])
    n = max_goals + 1
    joint = np.zeros((n, n))
    for (h, a), p in cs_probs.items():
        if h < n and a < n:
            joint[h, a] = p

    # Provisional lambdas from the enumerated cells, to shape the residual buckets.
    tot = joint.sum()
    if tot <= 0:
        raise ValueError("correct-score book contained no usable prices")
    lh = float((joint.sum(axis=1) * np.arange(n)).sum() / tot)
    la = float((joint.sum(axis=0) * np.arange(n)).sum() / tot)

    shape = np.outer(poisson.pmf(np.arange(n), max(lh, 0.1)),
                     poisson.pmf(np.arange(n), max(la, 0.1)))
    enumerated = joint > 0
    idx = np.arange(n)
    hw = (idx[:, None] > idx[None, :]) & ~enumerated
    aw = (idx[:, None] < idx[None, :]) & ~enumerated
    dr = (idx[:, None] == idx[None, :]) & ~enumerated

    for mask, bucket in ((hw, residual_home), (aw, residual_away), (dr, residual_draw)):
        w = shape[mask]
        if bucket > 0 and w.sum() > 0:
            joint[mask] = bucket * w / w.sum()

    joint = joint / joint.sum()
    lh = float((joint.sum(axis=1) * idx).sum())
    la = float((joint.sum(axis=0) * idx).sum())
    return ScorelineDistribution(lh, la, 0.0, joint, source, markets_used=["CORRECT_SCORE"])


def from_lambdas(lh: float, la: float, rho: float = -0.03,
                 source: str = "model:fallback") -> ScorelineDistribution:
    """Distribution from explicit lambdas - used by the non-exchange fallback."""
    return ScorelineDistribution(lh, la, rho, build_joint(lh, la, rho), source)
