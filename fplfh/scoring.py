"""FPL scoring rules.

The points-per-event table is read live from ``bootstrap-static``
(``game_config.scoring``) so it tracks rule changes automatically. The API does
*not* publish the thresholds and divisors (DefCon thresholds, saves-per-point,
goals-conceded divisor), so those live in config and were derived empirically
by boundary search over finished gameweeks' ``explain`` blocks.

``validate_against_api`` re-runs that check and will flag any drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ELEMENT_TYPE_TO_POS, params


@dataclass
class ScoringRules:
    goals: dict[str, float]
    assists: float
    clean_sheets: dict[str, float]
    conceded_pts: dict[str, float]
    conceded_per: int
    saves_pts: float
    saves_per: int
    defcon_pts: dict[str, float]
    defcon_thresholds: dict[str, int]
    yellow: float
    red: float
    own_goal: float
    pen_saved: float
    pen_missed: float
    short_play_pts: float
    long_play_pts: float
    long_play_min: int
    source: str = "config"
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls) -> "ScoringRules":
        s = params()["scoring"]
        return cls(
            goals=dict(s["goals_scored"]),
            assists=float(s["assists"]),
            clean_sheets=dict(s["clean_sheets"]),
            conceded_pts=dict(s["goals_conceded"]["pts"]),
            conceded_per=int(s["goals_conceded"]["per_n_conceded"]),
            saves_pts=float(s["saves"]["pts"]),
            saves_per=int(s["saves"]["per_n_saves"]),
            defcon_pts=dict(s["defensive_contribution"]["pts"]),
            defcon_thresholds=dict(s["defensive_contribution"]["thresholds"]),
            yellow=float(s["yellow_cards"]),
            red=float(s["red_cards"]),
            own_goal=float(s["own_goals"]),
            pen_saved=float(s["penalties_saved"]),
            pen_missed=float(s["penalties_missed"]),
            short_play_pts=float(s["minutes"]["short_play_pts"]),
            long_play_pts=float(s["minutes"]["long_play_pts"]),
            long_play_min=int(s["minutes"]["long_play_min"]),
        )

    @classmethod
    def from_bootstrap(cls, bootstrap: dict) -> "ScoringRules":
        """Live rules from the API, with config supplying the thresholds."""
        rules = cls.from_config()
        sc = bootstrap.get("game_config", {}).get("scoring")
        if not sc:
            rules.warnings.append(
                "bootstrap has no game_config.scoring; using config values only"
            )
            return rules
        rules.source = "bootstrap-static"

        def _cmp(name, live, cfg):
            if live != cfg:
                rules.warnings.append(f"{name}: API={live} config={cfg} (using API)")

        _cmp("goals_scored", sc["goals_scored"], rules.goals)
        rules.goals = dict(sc["goals_scored"])
        _cmp("assists", sc["assists"], rules.assists)
        rules.assists = float(sc["assists"])
        _cmp("clean_sheets", sc["clean_sheets"], rules.clean_sheets)
        rules.clean_sheets = dict(sc["clean_sheets"])
        _cmp("goals_conceded", sc["goals_conceded"], rules.conceded_pts)
        rules.conceded_pts = dict(sc["goals_conceded"])
        _cmp("defensive_contribution", sc["defensive_contribution"], rules.defcon_pts)
        rules.defcon_pts = dict(sc["defensive_contribution"])
        for key, attr in [
            ("saves", "saves_pts"), ("yellow_cards", "yellow"), ("red_cards", "red"),
            ("own_goals", "own_goal"), ("penalties_saved", "pen_saved"),
            ("penalties_missed", "pen_missed"), ("short_play", "short_play_pts"),
            ("long_play", "long_play_pts"),
        ]:
            if key in sc:
                _cmp(key, sc[key], getattr(rules, attr))
                setattr(rules, attr, float(sc[key]))
        return rules


def validate_against_api(client, events: list[int] | None = None) -> dict[str, Any]:
    """Re-derive thresholds and divisors from finished gameweeks.

    Scans the ``explain`` blocks and reports, per position, the smallest stat
    value that scored and the largest that did not. If the true threshold has
    moved, the two will straddle a different number than config says.
    """
    bootstrap = client.bootstrap()
    pos_of = {e["id"]: ELEMENT_TYPE_TO_POS[e["element_type"]] for e in bootstrap["elements"]}
    events = events or client.finished_events()
    rules = ScoringRules.from_config()

    scored: dict[tuple[str, str], list[int]] = {}
    missed: dict[tuple[str, str], list[int]] = {}
    for ev in events:
        for el in client.live(ev)["elements"]:
            pos = pos_of.get(el["id"])
            if pos is None:
                continue
            got = {s["identifier"]: s["points"] for fx in el["explain"] for s in fx["stats"]}
            stats = el["stats"]
            for ident in ("defensive_contribution", "saves"):
                val = stats.get(ident, 0)
                if val <= 0:
                    continue
                key = (pos, ident)
                (scored if got.get(ident, 0) > 0 else missed).setdefault(key, []).append(val)

    report: dict[str, Any] = {"events_scanned": events, "findings": {}, "ok": True}
    for pos, thr in rules.defcon_thresholds.items():
        key = (pos, "defensive_contribution")
        lo = min(scored.get(key, []), default=None)
        hi = max(missed.get(key, []), default=None)
        entry = {"config_threshold": thr, "min_scoring": lo, "max_non_scoring": hi}
        if lo is not None and lo != thr:
            entry["drift"] = f"observed minimum scoring value {lo} != config {thr}"
            report["ok"] = False
        elif lo is None:
            entry["note"] = f"no {pos} reached the threshold in the scanned gameweeks"
        report["findings"][f"defcon_{pos}"] = entry

    gk = ("GKP", "saves")
    lo = min(scored.get(gk, []), default=None)
    report["findings"]["saves_divisor"] = {
        "config_per_n": rules.saves_per,
        "min_saves_scoring": lo,
        "consistent": (lo is None or lo == rules.saves_per),
    }
    if lo is not None and lo != rules.saves_per:
        report["ok"] = False
    return report
