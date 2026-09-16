"""Free Hit squad selection as a mixed-integer program.

A Free Hit is the cleanest optimisation problem in FPL. The squad lasts exactly
one gameweek, so there is no transfer cost, no team value to protect and no
future fixtures to trade off - the objective collapses to "maximise expected
points this week subject to the squad rules". That makes an exact solve both
possible and worthwhile; greedy value-per-million heuristics leave real points
on the table because the budget and club limits interact.

Formulated with three sets of binaries:

    squad[i]    player i is one of the 15
    start[i]    player i is in the starting XI
    captain[i]  player i wears the armband

and solved exactly with CBC. Roughly 2,000 binaries, which CBC handles in
about a second.

The bench is worth a little, not nothing: a benched player scores when a
starter does not play, through autosubs. That is handled with a weight rather
than an explicit autosub simulation, because the weight is a second-order term
and modelling it exactly would make the program non-linear for very little
gain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from ..config import params


@dataclass
class Squad:
    """A solved 15-player squad."""

    players: pd.DataFrame           # all 15, with `is_starter` and `is_captain`
    total_xp: float                 # objective: XI + captain + weighted bench
    starting_xp: float              # XI only, captain doubled
    cost: float
    formation: str
    captain: str
    vice_captain: str
    bench: list[str] = field(default_factory=list)
    status: str = "Optimal"

    def __repr__(self) -> str:
        return (
            f"<Squad {self.formation} xP={self.starting_xp:.2f} "
            f"cost={self.cost:.1f}m captain={self.captain}>"
        )

    def summary(self) -> str:
        lines = [
            f"Formation {self.formation}   cost L{self.cost:.1f}m   "
            f"XI xP {self.starting_xp:.2f} (incl. captain)",
            f"Captain: {self.captain}   Vice: {self.vice_captain}",
            "",
        ]
        xi = self.players[self.players["is_starter"]].sort_values(
            ["position_order", "xp"], ascending=[True, False]
        )
        for _, r in xi.iterrows():
            mark = " (C)" if r["is_captain"] else ""
            lines.append(
                f"  {r['position']:>3}  {r['web_name'][:18]:<18} {r['team']:<4} "
                f"L{r['price']:>4.1f}  {r['opponent']:<8} xP {r['xp']:>5.2f}{mark}"
            )
        bench = self.players[~self.players["is_starter"]].sort_values(
            ["position_order", "xp"], ascending=[True, False]
        )
        lines.append("")
        lines.append("  Bench:")
        for _, r in bench.iterrows():
            lines.append(
                f"  {r['position']:>3}  {r['web_name'][:18]:<18} {r['team']:<4} "
                f"L{r['price']:>4.1f}  {r['opponent']:<8} xP {r['xp']:>5.2f}"
            )
        return "\n".join(lines)


POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def optimise_free_hit(
    players: pd.DataFrame,
    budget: float | None = None,
    xp_col: str = "xp",
    locked: list | None = None,
    banned: list | None = None,
    max_per_team: int | None = None,
    bench_weight: float | None = None,
    exclude_squads: list[list[int]] | None = None,
    diversity: int = 2,
    min_minutes_filter: float = 0.0,
    solver_msg: bool = False,
) -> Squad | None:
    """Solve for the optimal Free Hit squad.

    ``budget`` is in millions and defaults to the game's 100.0. On a real Free
    Hit the ceiling is your current squad value plus bank, so pass that.

    ``locked`` forces players in, ``banned`` forces them out - both accept
    player ids or web names.

    ``exclude_squads`` carries starting-XI player ids from previously returned
    squads. Each becomes a no-good cut requiring at least ``diversity`` changes
    to the XI, so repeated calls yield genuinely different teams rather than the
    same eleven behind a reshuffled bench.
    """
    cfg = params()
    sq = cfg["squad"]
    opt = cfg["optimiser"]

    budget_tenths = int(round((budget * 10) if budget is not None else sq["budget_tenths"]))
    max_per_team = max_per_team or int(sq["max_per_team"])
    bench_weight = float(opt["bench_weight"]) if bench_weight is None else bench_weight
    bench_gk_weight = float(opt["bench_gk_weight"])
    cap_mult = float(opt["captain_multiplier"])

    df = players.copy()
    if min_minutes_filter > 0:
        # Keep anyone locked in regardless of the filter.
        keep = df["xmins"] >= min_minutes_filter if "xmins" in df.columns else True
        df = df[keep]
    df = df.reset_index(drop=True)
    if df.empty:
        return None

    def _resolve(names) -> set[int]:
        out: set[int] = set()
        if not names:
            return out
        lookup = {str(r.web_name).lower(): int(r.player_id) for _, r in df.iterrows()}
        for n in names:
            if isinstance(n, (int,)) or str(n).isdigit():
                out.add(int(n))
            elif str(n).lower() in lookup:
                out.add(lookup[str(n).lower()])
        return out

    lock_ids, ban_ids = _resolve(locked), _resolve(banned)
    df = df[~df["player_id"].isin(ban_ids)].reset_index(drop=True)

    idx = list(df.index)
    xp = df[xp_col].fillna(0.0).to_dict()
    cost = df["cost_tenths"].to_dict()
    pos = df["position"].to_dict()
    team = df["team_id"].to_dict()
    pid = df["player_id"].to_dict()

    prob = pulp.LpProblem("free_hit", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    start = pulp.LpVariable.dicts("start", idx, cat="Binary")
    capt = pulp.LpVariable.dicts("capt", idx, cat="Binary")

    # Objective: starters at face value, captain counted a second time, bench
    # discounted to its autosub value.
    prob += pulp.lpSum(
        [start[i] * xp[i] for i in idx]
        + [capt[i] * xp[i] * (cap_mult - 1.0) for i in idx]
        + [
            (squad[i] - start[i])
            * xp[i]
            * (bench_gk_weight if pos[i] == "GKP" else bench_weight)
            for i in idx
        ]
    )

    # --- squad composition --------------------------------------------------
    prob += pulp.lpSum([squad[i] for i in idx]) == int(sq["size"])
    for p, n in sq["select"].items():
        prob += pulp.lpSum([squad[i] for i in idx if pos[i] == p]) == int(n)
    prob += pulp.lpSum([squad[i] * cost[i] for i in idx]) <= budget_tenths
    for t in set(team.values()):
        prob += pulp.lpSum([squad[i] for i in idx if team[i] == t]) <= max_per_team

    # --- starting XI --------------------------------------------------------
    prob += pulp.lpSum([start[i] for i in idx]) == int(sq["starting"])
    for i in idx:
        prob += start[i] <= squad[i]
    for p in ("GKP", "DEF", "MID", "FWD"):
        sel = [start[i] for i in idx if pos[i] == p]
        prob += pulp.lpSum(sel) >= int(sq["min_play"][p])
        prob += pulp.lpSum(sel) <= int(sq["max_play"][p])

    # --- captain ------------------------------------------------------------
    prob += pulp.lpSum([capt[i] for i in idx]) == 1
    for i in idx:
        prob += capt[i] <= start[i]

    # --- user constraints ---------------------------------------------------
    for i in idx:
        if pid[i] in lock_ids:
            prob += squad[i] == 1

    # --- no-good cuts for alternative squads --------------------------------
    # The cut applies to the starting XI, not the 15. Cutting on the full squad
    # looks right but is nearly useless: the cheapest way to satisfy it is to
    # swap one *bench* player, so the solver returns an identical XI and an
    # identical score, and the "alternatives" are alternatives in name only.
    for prev in exclude_squads or []:
        prev_idx = [i for i in idx if pid[i] in set(prev)]
        if prev_idx:
            prob += pulp.lpSum([start[i] for i in prev_idx]) <= max(
                len(prev_idx) - max(int(diversity), 1), 0
            )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0))
    if pulp.LpStatus[status] != "Optimal":
        return None

    chosen = [i for i in idx if squad[i].value() and squad[i].value() > 0.5]
    if not chosen:
        return None
    starters = {i for i in chosen if start[i].value() and start[i].value() > 0.5}
    captain_i = next((i for i in chosen if capt[i].value() and capt[i].value() > 0.5), None)

    out = df.loc[chosen].copy()
    out["is_starter"] = out.index.isin(starters)
    out["is_captain"] = out.index == captain_i
    out["position_order"] = out["position"].map(POSITION_ORDER)

    xi = out[out["is_starter"]]
    counts = xi["position"].value_counts()
    formation = f"{counts.get('DEF',0)}-{counts.get('MID',0)}-{counts.get('FWD',0)}"
    starting_xp = float(xi[xp_col].sum() + (xp[captain_i] if captain_i is not None else 0.0))

    # Vice-captain: the best starter who is not the captain. It only pays out
    # if the captain does not appear, so it is reported rather than optimised.
    vc_pool = xi[~xi["is_captain"]].sort_values(xp_col, ascending=False)
    vice = str(vc_pool.iloc[0]["web_name"]) if len(vc_pool) else "-"

    return Squad(
        players=out.sort_values(["position_order", "xp"], ascending=[True, False]),
        total_xp=float(pulp.value(prob.objective)),
        starting_xp=starting_xp,
        cost=float(out["cost_tenths"].sum() / 10.0),
        formation=formation,
        captain=str(out.loc[captain_i, "web_name"]) if captain_i is not None else "-",
        vice_captain=vice,
        bench=[str(r["web_name"]) for _, r in out[~out["is_starter"]].iterrows()],
        status=pulp.LpStatus[status],
    )


def top_squads(players: pd.DataFrame, n: int = 3, **kwargs) -> list[Squad]:
    """The best ``n`` genuinely distinct squads.

    Useful because the optimal squad is often a fraction of a point ahead of
    several alternatives with quite different risk profiles - worth seeing
    before committing a chip. ``diversity`` sets how many XI places must change
    between them.
    """
    out: list[Squad] = []
    exclude: list[list[int]] = list(kwargs.pop("exclude_squads", []) or [])
    for _ in range(n):
        s = optimise_free_hit(players, exclude_squads=exclude, **kwargs)
        if s is None:
            break
        out.append(s)
        exclude.append([int(p) for p in s.players.loc[s.players["is_starter"], "player_id"]])
    return out


VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 2, 3), (5, 3, 2), (5, 4, 1),
]


def best_xi(squad: pd.DataFrame, xp_col: str = "xp") -> Squad:
    """Best starting XI, captain and bench order from a squad you already own.

    Unlike ``optimise_free_hit`` this chooses nothing about *which* players you
    have - only how to line them up. There are just eight legal formations, so
    it enumerates rather than solving a program.

    The bench is ordered by expected points, which is how automatic
    substitutions should be prioritised.
    """
    df = squad.copy().reset_index(drop=True)
    gks = df[df["position"] == "GKP"].sort_values(xp_col, ascending=False)
    if gks.empty:
        raise ValueError("squad contains no goalkeeper")

    best = None
    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        picks = [gks.index[0]]
        ok = True
        for pos, n in (("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)):
            avail = df[df["position"] == pos].sort_values(xp_col, ascending=False)
            if len(avail) < n:
                ok = False
                break
            picks.extend(avail.index[:n])
        if not ok:
            continue
        total = float(df.loc[picks, xp_col].sum())
        if best is None or total > best[0]:
            best = (total, picks, f"{n_def}-{n_mid}-{n_fwd}")
    if best is None:
        raise ValueError("no legal formation available from this squad")

    total, picks, formation = best
    df["is_starter"] = df.index.isin(picks)
    df["position_order"] = df["position"].map(POSITION_ORDER)

    xi = df[df["is_starter"]].sort_values(xp_col, ascending=False)
    cap_idx = xi.index[0]
    df["is_captain"] = df.index == cap_idx
    vice = str(xi.iloc[1]["web_name"]) if len(xi) > 1 else "-"

    bench = df[~df["is_starter"]].sort_values(
        ["position_order", xp_col], ascending=[True, False]
    )
    return Squad(
        players=df.sort_values(["position_order", xp_col], ascending=[True, False]),
        total_xp=total + float(df.loc[cap_idx, xp_col]),
        starting_xp=total + float(df.loc[cap_idx, xp_col]),
        cost=float(df["cost_tenths"].sum() / 10.0) if "cost_tenths" in df else 0.0,
        formation=formation,
        captain=str(df.loc[cap_idx, "web_name"]),
        vice_captain=vice,
        bench=[str(r["web_name"]) for _, r in bench.iterrows()],
    )
