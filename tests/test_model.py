"""Invariants that must hold for the projections to mean anything.

Run with pytest, or directly:  python tests/test_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from fplfh.model.events import (
    defcon_probability,
    distribute_team_goals,
    goals_from_anytime_market,
    expected_save_points,
)
from fplfh.model.scoreline import build_joint, fit_to_markets, from_lambdas
from fplfh.optimise import optimise_free_hit


# ---------------------------------------------------------------- scoreline
def test_joint_is_a_distribution():
    j = build_joint(1.6, 1.2, -0.03)
    assert abs(j.sum() - 1.0) < 1e-9
    assert (j >= 0).all()


def test_fit_recovers_known_lambdas():
    """Round-trip: generate market probabilities from known lambdas, refit."""
    truth = from_lambdas(2.1, 0.9, rho=-0.03)
    got = fit_to_markets(
        p_home=truth.p_win(True),
        p_draw=truth.p_draw(),
        p_away=truth.p_win(False),
        over_lines={1.5: truth.p_over(1.5), 2.5: truth.p_over(2.5), 3.5: truth.p_over(3.5)},
        p_btts=truth.p_btts(),
    )
    assert abs(got.lambda_home - 2.1) < 0.05, got.lambda_home
    assert abs(got.lambda_away - 0.9) < 0.05, got.lambda_away
    assert got.fit_residual < 0.01


def test_conceded_floor_is_concave():
    """E[floor(GC/2)] must sit strictly below E[GC]/2.

    Guards the single largest arithmetic trap in the model: using E[GC]/2
    overstates the deduction by ~100% for a good defence.
    """
    d = from_lambdas(2.4, 0.8)
    for is_home in (True, False):
        exact = d.expected_conceded_penalty_units(is_home, 2)
        naive = d.expected_conceded(is_home) / 2
        assert exact < naive, (exact, naive)


def test_clean_sheet_matches_marginal():
    d = from_lambdas(1.7, 1.1)
    assert abs(d.p_clean_sheet(True) - d.goals_against(True)[0]) < 1e-12
    # Stronger attack faced => lower clean sheet probability.
    assert d.p_clean_sheet(False) < d.p_clean_sheet(True)


def test_result_probabilities_sum_to_one():
    d = from_lambdas(1.5, 1.3)
    assert abs(d.p_win(True) + d.p_draw() + d.p_win(False) - 1.0) < 1e-9


# ------------------------------------------------------- price -> probability
def test_back_lay_midpoint_is_taken_in_probability_space():
    """An exchange quotes two sides; the fair value sits between them.

    Measured on live data: back overround 1.0047, lay 0.9926. Using the back
    side alone overstates every probability by roughly half a percent before
    normalisation, so the midpoint matters.
    """
    from fplfh.odds.oddsapi import _two_sided_h2h

    book = {"markets": [
        {"key": "h2h", "outcomes": [{"name": "A", "price": 2.00},
                                    {"name": "Draw", "price": 3.80},
                                    {"name": "B", "price": 4.20}]},
        {"key": "h2h_lay", "outcomes": [{"name": "A", "price": 2.02},
                                        {"name": "Draw", "price": 3.90},
                                        {"name": "B", "price": 4.30}]},
    ]}
    probs, side = _two_sided_h2h(book)
    assert side == "back+lay mid"
    assert abs(sum(probs.values()) - 1.0) < 1e-9

    # The midpoint must sit between the two one-sided books - compared *after*
    # normalisation, since normalising rescales the whole book and a normalised
    # value need not lie between the raw one-sided prices.
    back_only, side2 = _two_sided_h2h({"markets": [book["markets"][0]]})
    assert side2 == "back-only"
    lay_as_book = {"markets": [{"key": "h2h", "outcomes": book["markets"][1]["outcomes"]}]}
    lay_only, _ = _two_sided_h2h(lay_as_book)

    lo, hi = sorted((back_only["A"], lay_only["A"]))
    assert lo < probs["A"] < hi, (lo, probs["A"], hi)


def test_h2h_falls_back_to_back_only_without_lay():
    from fplfh.odds.oddsapi import _two_sided_h2h

    probs, side = _two_sided_h2h({"markets": [
        {"key": "h2h", "outcomes": [{"name": "A", "price": 2.0},
                                    {"name": "Draw", "price": 3.5},
                                    {"name": "B", "price": 4.0}]}]})
    assert side == "back-only"
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_totals_consensus_devigs_each_source_then_medians():
    """Totals come from bookmakers, which DO carry a margin.

    Each source's two-runner book is normalised individually before the median
    is taken - normalising after averaging would leave the margin in.
    """
    from fplfh.odds.oddsapi import _totals_consensus

    def src(key, over, under):
        return {"key": key, "markets": [{"key": "totals", "outcomes": [
            {"name": "Over", "price": over, "point": 2.5},
            {"name": "Under", "price": under, "point": 2.5}]}]}

    # Three books, each ~5% overround, agreeing on a fair over-2.5 near 0.55.
    event = {"bookmakers": [src("a", 1.80, 2.10), src("b", 1.82, 2.08),
                            src("c", 1.78, 2.12)]}
    lines, n, sources = _totals_consensus(event)
    assert set(sources) == {"a", "b", "c"} and n == 3
    assert 0.50 < lines[2.5] < 0.60, lines
    # De-vigged, so strictly below the raw implied probability of the Over price.
    assert lines[2.5] < 1 / 1.80


def test_markets_without_totals_are_flagged():
    """A 1X2-only fit is exactly determined and must say so.

    Three outcomes for three free parameters leaves the goal *total* pinned by
    the rho prior rather than by data, and the residual carries no information.
    """
    from fplfh.odds.oddsapi import scoreline_from_event

    ev = _odds_event()
    ev["bookmakers"][0]["markets"] = [
        m for m in ev["bookmakers"][0]["markets"] if m["key"] != "totals"
    ]
    dist, note = scoreline_from_event(ev, "Brighton", "Arsenal")
    assert dist is not None
    assert "NO TOTALS" in note, note


# ------------------------------------------------------------------- events
def test_goal_shares_sum_to_team_lambda():
    """The anchoring property: player goals must reconcile to the team total."""
    w = np.array([0.6, 0.4, 0.25, 0.1, 0.05])
    out = distribute_team_goals(w, team_lambda=2.0, own_goal_fraction=0.0)
    assert abs(out.sum() - 2.0) < 1e-9


def test_anytime_market_inversion_beats_raw_probability():
    """mu = -ln(1-p) must exceed p for a heavy favourite."""
    probs = np.array([0.60, 0.30, 0.15, 0.10, 0.05])
    out = goals_from_anytime_market(probs, team_lambda=float(-np.log(1 - probs).sum()),
                                    own_goal_fraction=0.0)
    assert out[0] > probs[0] * 1.3


def test_defcon_monotonic_in_rate_and_opponent():
    rate = np.array([4.0, 8.0, 12.0])
    p = defcon_probability(rate, np.full(3, 90.0), "DEF", np.full(3, 1.45))
    assert p[0] < p[1] < p[2]
    # A side expected to concede more does more defending.
    low = defcon_probability(np.array([8.0]), np.array([90.0]), "DEF", np.array([0.8]))
    high = defcon_probability(np.array([8.0]), np.array([90.0]), "DEF", np.array([2.2]))
    assert high[0] > low[0]


def test_defcon_negbin_exceeds_poisson():
    """Overdispersion must raise the tail probability, or the fit is pointless."""
    from scipy.stats import poisson
    mu = 8.0
    nb = defcon_probability(np.array([mu]), np.array([90.0]), "DEF", np.array([1.45]))[0]
    po = poisson.sf(9, mu)
    assert nb > po


def test_threshold_averages_over_minutes_not_expected_minutes():
    """P(X >= threshold) must be averaged over the minutes distribution.

    Plugging expected minutes into a threshold probability computes f(E[mins])
    instead of E[f(mins)], and the two differ sharply because the tail is convex
    in minutes. A player with a 40% chance of playing 85 minutes is not the same
    as one certain to play 34: the first can reach 10 defensive actions, the
    second essentially cannot. Backtesting caught this understating league-wide
    DefCon hits by 59%.
    """
    from fplfh.model.events import APPEARANCE_MODES as AM
    from fplfh.model.events import minutes_mixture

    rate = np.array([9.0])
    p_60, p_start, p_play = np.array([0.4]), np.array([0.45]), np.array([0.5])
    # Expected minutes implied by that mixture. Derived from the shared
    # constants rather than hardcoded, so refitting them cannot silently
    # invalidate this test.
    xmins = np.array([
        0.40 * AM["full"] + 0.05 * AM["early_sub"] + 0.05 * AM["cameo"]
    ])
    opp = np.array([1.45])

    point = defcon_probability(rate, xmins, "DEF", opp)[0]
    mixed = defcon_probability(
        rate, xmins, "DEF", opp, mix=minutes_mixture(p_60, p_start, p_play)
    )[0]
    assert mixed > point * 1.5, (mixed, point)

    # A certain full-match starter must be unaffected by the treatment, since
    # there is no minutes uncertainty left to average over. Evaluated at the
    # shared constant, not a hardcoded 85.
    one = np.array([1.0])
    full = np.array([AM["full"]])
    pt = defcon_probability(rate, full, "DEF", opp)[0]
    mx = defcon_probability(rate, full, "DEF", opp,
                            mix=minutes_mixture(one, one, one))[0]
    assert abs(mx - pt) < 1e-9


def test_saves_intercept_matters():
    """A keeper in a quiet game still makes saves.

    The affine fit and a through-the-origin fit agree on busy fixtures but
    diverge sharply on quiet ones - by ~12x at an opponent lambda of 0.4. That
    gap is exactly the keepers behind good defences, so getting it wrong would
    systematically misprice the cheapest useful GK picks.
    """
    from scipy.stats import poisson

    quiet = expected_save_points(np.array([0.4]), np.array([90.0]))[0]
    assert quiet > 0.25

    ks = np.arange(25)
    through_origin = float((poisson.pmf(ks, 1.523 * 0.4) * np.floor(ks / 3)).sum())
    assert quiet > 5 * through_origin

    # And it must stay monotonic in opponent strength.
    busy = expected_save_points(np.array([2.5]), np.array([90.0]))[0]
    assert busy > quiet


# ------------------------------------------------------------- odds adapter
def _odds_event(home="Brighton and Hove Albion", away="Arsenal",
                ph=0.45, pdw=0.26, pa=0.29, over=0.55):
    px = lambda p: round(1.0 / p, 3)
    return {
        "id": "ev1", "home_team": home, "away_team": away,
        "commence_time": "2026-09-19T14:00:00Z",
        "bookmakers": [{"key": "betfair_ex_uk", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, "price": px(ph)},
                                        {"name": away, "price": px(pa)},
                                        {"name": "Draw", "price": px(pdw)}]},
            {"key": "totals", "outcomes": [
                {"name": "Over", "price": px(over), "point": 2.5},
                {"name": "Under", "price": px(1 - over), "point": 2.5}]}]}]}


def test_odds_adapter_recovers_input_probabilities():
    """A scoreline fitted from odds must reproduce the odds it was fitted to."""
    from fplfh.odds.oddsapi import scoreline_from_event

    dist, note = scoreline_from_event(_odds_event(), "Brighton", "Arsenal")
    assert dist is not None, note
    assert abs(dist.p_win(True) - 0.45) < 0.01
    assert abs(dist.p_over(2.5) - 0.55) < 0.01
    assert dist.source.startswith("oddsapi:betfair_ex_uk")


def test_odds_team_aliases_resolve():
    """FPL and provider spellings of the same club must join.

    The join is the most fragile part of any provider integration, and a silent
    miss degrades to the fallback model while still looking market-derived.
    """
    from fplfh.naming import team_key

    for fpl_name, provider_name in [
        ("Spurs", "Tottenham Hotspur"), ("Man Utd", "Manchester United"),
        ("Man City", "Manchester City"), ("Nott'm Forest", "Nottingham Forest"),
        ("Brighton", "Brighton and Hove Albion"), ("Newcastle", "Newcastle United"),
    ]:
        assert team_key(fpl_name) & team_key(provider_name), (fpl_name, provider_name)

    # And must NOT collide across distinct clubs.
    assert not (team_key("Man Utd") & team_key("Man City"))
    assert not (team_key("Leeds") & team_key("Leicester"))


def test_card_rate_is_shrunk():
    """A card in a cameo must not imply an absurd per-90 rate.

    Bookings are rare, so an unshrunk rate is almost entirely noise: one player
    showed 18.0 yellows per 90 off a single card in five minutes, and such
    players supplied 18% of all card deductions league-wide.
    """
    from fplfh.model.events import shrink_rate

    base, k = 0.188, 30.0
    # One card in five minutes -> a nonsense raw rate.
    raw = 1.0 / (5.0 / 90.0)
    assert raw > 17
    shrunk = shrink_rate(np.array([raw]), np.array([5.0]), base, 90.0 * k)[0]
    assert shrunk < 0.30, shrunk

    # An established player keeps most of their own signal.
    heavy = shrink_rate(np.array([0.40]), np.array([90.0 * 30]), base, 90.0 * k)[0]
    assert heavy > 0.28, heavy


def test_penalty_deduction_scales_with_minutes():
    """A taker who will not play cannot miss a penalty.

    Every other component is scaled by minutes; this one was not, so injured
    penalty takers received a deduction for matches they never appeared in and
    ended up with a negative projected total made of nothing else.
    """
    from fplfh.model.events import expected_penalty_points

    taker, lam = np.array([True]), np.array([1.8])

    out = expected_penalty_points(taker, lam, np.array([0.0]))
    assert out[0] == 0.0, out

    full = expected_penalty_points(taker, lam, np.array([90.0]))[0]
    half = expected_penalty_points(taker, lam, np.array([45.0]))[0]
    assert full < 0 and abs(half - full / 2) < 1e-12

    # A non-taker is never deducted, however long they play.
    assert expected_penalty_points(np.array([False]), lam, np.array([90.0]))[0] == 0.0


def test_team_start_probabilities_sum_to_eleven():
    """A team starts exactly 11 players - a hard constraint the per-player
    model cannot see.

    Shrinkage moves each estimate on its own merits, so team totals drift.
    Measured at 9.85 before normalisation, which understated every clean sheet
    and appearance point by the same proportion.
    """
    from fplfh.model.minutes import _normalise_team_starts

    df = pd.DataFrame({
        "team_id": [1] * 20 + [2] * 20,
        # Team 1 starts far too few; team 2 far too many.
        "p_start": [0.4] * 20 + [0.8] * 20,
    })
    out = _normalise_team_starts(df.copy(), target=11.0, ceiling=0.97)
    for _, g in out.groupby("team_id"):
        assert abs(g["p_start"].sum() - 11.0) < 1e-6, g["p_start"].sum()
    assert (out["p_start"] <= 0.97 + 1e-9).all()

    # A player at zero (injured) must stay at zero - their share passes to
    # whoever plays instead, rather than being handed back to them.
    df2 = pd.DataFrame({"team_id": [1] * 20,
                        "p_start": [0.0] + [0.6] * 19})
    out2 = _normalise_team_starts(df2.copy(), target=11.0, ceiling=0.97)
    assert out2["p_start"].iloc[0] == 0.0
    assert abs(out2["p_start"].sum() - 11.0) < 1e-6


def test_normalise_handles_atomic_unicode_letters():
    """Letters that are atomic in Unicode, not base+accent, need transliterating.

    NFKD decomposition cannot help with these, and the ASCII filter would
    otherwise delete them outright - turning "Hojlund" into "h jlund", where the
    stripped letter becomes a space and splits the name in two.
    """
    from fplfh.naming import normalise

    for typed, actual in [
        ("Odegaard", "Ødegaard"), ("Hojlund", "Højlund"),
        ("Gross", "Groß"), ("Norgaard", "Nørgaard"),
        ("Duric", "Đurić"), ("Sorensen", "Sørensen"),
        ("Guehi", "Guéhi"), ("Sesko", "Šeško"),
    ]:
        assert normalise(typed) == normalise(actual), (typed, actual,
                                                       normalise(typed), normalise(actual))
    # And no name may normalise to something containing a stray space where the
    # original had a single token.
    assert " " not in normalise("Højlund")


def test_best_xi_picks_a_legal_formation():
    """Choosing how to line up an existing squad, not which players to own."""
    from fplfh.optimise import best_xi

    rows = []
    for pos, n in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for j in range(n):
            rows.append({"player_id": len(rows), "web_name": f"{pos}{j}",
                         "position": pos, "team": "T", "team_id": len(rows) % 4,
                         "price": 5.0, "cost_tenths": 50, "opponent": "X",
                         "xp": 10.0 - len(rows)})
    s = best_xi(pd.DataFrame(rows))
    xi = s.players[s.players["is_starter"]]
    assert len(xi) == 11
    counts = xi["position"].value_counts()
    assert counts.get("GKP", 0) == 1
    assert 3 <= counts.get("DEF", 0) <= 5
    assert 2 <= counts.get("MID", 0) <= 5
    assert 1 <= counts.get("FWD", 0) <= 3
    # The captain must be the highest-scoring starter.
    assert s.captain == xi.nlargest(1, "xp").iloc[0]["web_name"]


# ------------------------------------------------------------ fixture outlook
def _outlook(rows):
    """rows: (team, event, xg_for, p_clean_sheet)"""
    return pd.DataFrame([
        {"team": t, "team_id": 0, "event": e, "opponent": "X", "is_home": True,
         "venue": "H", "xg_for": xg, "xg_against": 1.0, "p_clean_sheet": cs,
         "p_win": 0.4, "source": "model", "horizon": e}
        for t, e, xg, cs in rows
    ])


def test_blank_gameweeks_are_penalised_not_flattered():
    """A team with a blank must score below an identical team without one.

    Averaging over fixtures played would *reward* a blank, since the remaining
    games raise the mean. The weight of the whole window has to be charged
    whether or not a fixture exists.
    """
    from fplfh.outlook import rank_fixtures

    full = [("FULL", e, 2.0, 0.3) for e in range(1, 6)]
    blank = [("BLANK", e, 2.0, 0.3) for e in range(1, 6) if e != 3]
    r = rank_fixtures(_outlook(full + blank), window=5, discount=1.0, metric="attack")
    assert r.loc["BLANK", "blanks"] == 1
    assert r.loc["FULL", "score"] > r.loc["BLANK", "score"], r


def test_discount_shifts_weight_toward_near_fixtures():
    """Two teams, same total quality, opposite ordering."""
    from fplfh.outlook import rank_fixtures

    early = [("EARLY", e, 3.0 if e <= 2 else 1.0, 0.3) for e in range(1, 6)]
    late = [("LATE", e, 1.0 if e <= 2 else 3.0 - 0.5, 0.3) for e in range(1, 6)]
    rows = _outlook(early + late)

    flat = rank_fixtures(rows, window=5, discount=1.0, metric="attack")
    near = rank_fixtures(rows, window=5, discount=0.5, metric="attack")
    # Discounting hard must favour the team whose good games come first.
    assert (near.loc["EARLY", "score"] - near.loc["LATE", "score"]) >            (flat.loc["EARLY", "score"] - flat.loc["LATE", "score"])


def test_doubles_are_counted_and_add_up():
    from fplfh.outlook import rank_fixtures

    single = [("ONE", e, 1.5, 0.3) for e in range(1, 4)]
    double = [("TWO", e, 1.5, 0.3) for e in range(1, 4)] + [("TWO", 2, 1.5, 0.3)]
    r = rank_fixtures(_outlook(single + double), window=3, discount=1.0, metric="attack")
    assert r.loc["TWO", "doubles"] == 1 and r.loc["ONE", "doubles"] == 0
    assert r.loc["TWO", "score"] > r.loc["ONE", "score"]


# ------------------------------------------------------- horizon & transfers
def test_return_date_parsing():
    """Only a minority of flagged players state a date; departures never do."""
    from datetime import date
    from fplfh.availability import has_left, parse_return_date

    today = date(2026, 9, 16)
    assert parse_return_date("Expected back 11 Oct", today) == date(2026, 10, 11)
    assert parse_return_date("Suspended until 17 Oct", today) == date(2026, 10, 17)
    # A season spans two calendar years, so January rolls forward.
    assert parse_return_date("Expected back 3 Jan", today) == date(2027, 1, 3)
    assert parse_return_date("Knee injury - Unknown return date", today) is None
    assert parse_return_date("", today) is None
    assert parse_return_date(None, today) is None

    # A player who has left the league must never be restored, whatever the text.
    assert has_left("Has joined Al Hilal permanently")
    assert parse_return_date("Has joined Barcelona permanently", today) is None


def test_transfer_plan_is_executable():
    """A plan must be self-consistent, not just a list of good-looking swaps.

    Three ways a naive greedy list goes wrong: it buys the same bargain for
    several outgoing players, it ignores that earlier swaps move the bank, and
    it can breach the three-per-club limit partway through.
    """
    from fplfh.horizon import transfer_summary

    sug = pd.DataFrame([
        # Everyone wants the same cheap upgrade.
        {"out": "A", "out_team": "T1", "out_price": 5.0, "out_xp": 1.0,
         "in": "Z", "in_team": "T9", "in_price": 5.0, "in_xp": 9.0,
         "gain": 8.0, "cost": 0.0, "in_minutes_basis": "observed"},
        {"out": "B", "out_team": "T2", "out_price": 5.0, "out_xp": 2.0,
         "in": "Z", "in_team": "T9", "in_price": 5.0, "in_xp": 9.0,
         "gain": 7.0, "cost": 0.0, "in_minutes_basis": "observed"},
        # Affordable alone, but not after the first swap spends the bank.
        {"out": "B", "out_team": "T2", "out_price": 5.0, "out_xp": 2.0,
         "in": "Y", "in_team": "T8", "in_price": 9.0, "in_xp": 8.0,
         "gain": 6.0, "cost": 4.0, "in_minutes_basis": "observed"},
    ])
    plan = transfer_summary(sug, bank=0.5, squad_teams=["T1", "T2"], free_transfers=1)

    assert plan["in"].is_unique and plan["out"].is_unique
    assert (plan["bank_after"] >= -1e-9).all(), plan
    # The 4.0m move is unaffordable from a 0.5m bank and must be dropped.
    assert "Y" not in set(plan["in"])


def test_transfer_plan_respects_club_limit():
    from fplfh.horizon import transfer_summary

    # Squad already has three from T9; buying a fourth is illegal.
    sug = pd.DataFrame([
        {"out": "A", "out_team": "T1", "out_price": 5.0, "out_xp": 1.0,
         "in": "Z", "in_team": "T9", "in_price": 5.0, "in_xp": 9.0,
         "gain": 8.0, "cost": 0.0, "in_minutes_basis": "observed"},
    ])
    plan = transfer_summary(sug, bank=5.0, squad_teams=["T1", "T9", "T9", "T9"],
                            free_transfers=1, max_per_team=3)
    assert plan.empty, plan


def test_hit_cost_applies_beyond_free_transfers():
    from fplfh.horizon import transfer_summary

    sug = pd.DataFrame([
        {"out": c, "out_team": f"T{i}", "out_price": 5.0, "out_xp": 1.0,
         "in": c.lower(), "in_team": f"U{i}", "in_price": 5.0, "in_xp": 6.0,
         "gain": 5.0, "cost": 0.0, "in_minutes_basis": "observed"}
        for i, c in enumerate("ABC")
    ])
    plan = transfer_summary(sug, bank=0.0, squad_teams=["T0", "T1", "T2"],
                            free_transfers=1)
    assert list(plan["hit"]) == [0.0, 4.0, 4.0]
    assert list(plan["net"]) == [5.0, 1.0, 1.0]
    assert plan["cumulative_net"].iloc[-1] == 7.0


# ---------------------------------------------------------------- optimiser
def _fake_players(n=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pos = ["GKP", "DEF", "MID", "FWD"][i % 4]
        rows.append({
            "player_id": i, "web_name": f"P{i}", "full_name": f"Player {i}",
            "team_id": i % 20, "team": f"T{i % 20}", "position": pos,
            "cost_tenths": int(rng.integers(38, 96)),
            "opponent": "X", "xp": float(rng.gamma(2.0, 1.2)), "xmins": 80.0,
        })
    df = pd.DataFrame(rows)
    df["price"] = df["cost_tenths"] / 10.0
    return df


def test_optimiser_respects_every_constraint():
    s = optimise_free_hit(_fake_players(), budget=100.0)
    assert s is not None
    p = s.players
    assert len(p) == 15
    assert p["position"].value_counts().to_dict() == {"DEF": 5, "MID": 5, "GKP": 2, "FWD": 3}
    assert p["cost_tenths"].sum() <= 1000
    assert p["team_id"].value_counts().max() <= 3
    xi = p[p["is_starter"]]
    assert len(xi) == 11
    counts = xi["position"].value_counts()
    assert counts.get("GKP", 0) == 1
    assert 3 <= counts.get("DEF", 0) <= 5
    assert 2 <= counts.get("MID", 0) <= 5
    assert 1 <= counts.get("FWD", 0) <= 3
    assert int(p["is_captain"].sum()) == 1
    # Captain must be a starter, and the best one.
    cap = p[p["is_captain"]].iloc[0]
    assert bool(cap["is_starter"]) and cap["xp"] == xi["xp"].max()


def test_locked_and_banned_are_honoured():
    df = _fake_players()
    lock, ban = int(df.iloc[7].player_id), int(df.iloc[11].player_id)
    s = optimise_free_hit(df, budget=100.0, locked=[lock], banned=[ban])
    ids = set(s.players["player_id"])
    assert lock in ids and ban not in ids


def test_alternatives_differ_in_the_starting_xi():
    """The no-good cut must bite on the XI, not just the bench."""
    from fplfh.optimise import top_squads
    squads = top_squads(_fake_players(), n=3, budget=100.0, diversity=2)
    assert len(squads) == 3
    xis = [frozenset(s.players.loc[s.players["is_starter"], "player_id"]) for s in squads]
    assert len(set(xis)) == 3
    for a, b in zip(xis, xis[1:]):
        assert len(a - b) >= 2


def test_budget_binds():
    """A tighter budget must cost points, and must never be exceeded."""
    poor = optimise_free_hit(_fake_players(), budget=75.0)
    rich = optimise_free_hit(_fake_players(), budget=100.0)
    assert poor is not None and rich is not None
    assert poor.players["cost_tenths"].sum() <= 750
    assert rich.players["cost_tenths"].sum() <= 1000
    assert rich.starting_xp > poor.starting_xp


def test_infeasible_budget_returns_none():
    """An impossible budget must fail loudly-as-None, not return a bad squad."""
    assert optimise_free_hit(_fake_players(), budget=20.0) is None


def test_bench_probability_rises_with_standing():
    """A fringe player is far less likely to be subbed on than a rotation one.

    The flat constant this replaced had it backwards, treating bench chance as
    leftover probability and so handing the most weight to the player furthest
    from the team.
    """
    from fplfh.model.minutes import bench_probability

    avail = np.ones(4)
    p = bench_probability(np.array([0.00, 0.03, 0.30, 0.70]), avail)
    assert np.all(np.diff(p) > 0), f"not monotone in standing: {p}"
    # A player who never starts should be near-absent from the bench too.
    assert p[1] < 0.15, f"fringe player over-weighted: {p[1]:.3f}"
    # A rotation player sits on the fitted plateau.
    assert 0.30 < p[3] < 0.50, f"rotation player off the plateau: {p[3]:.3f}"
    # Unavailable means unavailable, on the bench as much as in the XI.
    assert bench_probability(np.array([0.5]), np.array([0.0]))[0] == 0.0


def test_team_sub_appearances_sum_to_target():
    """Each team's expected substitute appearances must match observation."""
    from fplfh.config import params
    from fplfh.model.minutes import estimate_minutes

    rng = np.random.default_rng(7)
    rows = []
    for team in range(1, 4):
        for i in range(25):
            rows.append({
                "player_id": team * 100 + i, "team_id": team, "team": f"T{team}",
                "web_name": f"p{team}_{i}", "position": ["GKP", "DEF", "MID", "FWD"][i % 4],
                "price": float(rng.uniform(4.0, 12.0)), "status": "a", "news": "",
                "chance_of_playing": None, "minutes": 0,
            })
    players = pd.DataFrame(rows)
    hist = pd.DataFrame([
        {"player_id": r["player_id"], "event": e,
         "minutes": int(rng.integers(0, 91)), "started": int(rng.random() < 0.4)}
        for r in rows for e in range(1, 6)
    ])
    out = estimate_minutes(players, hist, 6, apply_overrides=False)

    target = float(params()["minutes"]["subs_per_team"])
    subs = (out["p_play"] - out["p_start"]).clip(lower=0.0)
    per_team = subs.groupby(out["team_id"]).sum()
    assert np.allclose(per_team, target, atol=0.01), f"sub totals adrift: {per_team.to_dict()}"


def test_clean_sheet_uses_on_pitch_window_not_full_match():
    """The rule is 'no goal conceded while on the pitch', not 'team clean sheet'.

    A player substituted after 60 minutes still scores if the goal arrives after
    he leaves, so the probability that pays must exceed the team's 90-minute one.
    """
    from fplfh.model.events import on_pitch_clean_sheet, APPEARANCE_MODES

    p_cs = np.array([0.10, 0.25, 0.60])
    got = on_pitch_clean_sheet(p_cs, np.ones(3))
    assert np.all(got > p_cs), f"on-pitch window not applied: {got} vs {p_cs}"

    # Exact under Poisson: p ** (m / 90).
    expected = p_cs ** (APPEARANCE_MODES["full"] / 90.0)
    assert np.allclose(got, expected)

    # A full 90 minutes must reproduce the team figure exactly.
    assert np.allclose(on_pitch_clean_sheet(p_cs, np.ones(3), minutes_60_plus=90.0), p_cs)

    # Still gated on reaching 60 minutes.
    assert on_pitch_clean_sheet(np.array([0.5]), np.array([0.0]))[0] == 0.0


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
