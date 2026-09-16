"""Re-fit the empirical parameters in config/model_params.yaml.

Every non-rule number the model uses was estimated from data, and this script
reproduces those estimates so they can be refreshed as a season progresses
rather than ageing quietly in a config file.

Fits, in order:
  * DefCon negative-binomial dispersion, solved per position so that the
    modelled threshold hit-rate reproduces the observed one
  * goalkeeper saves as an affine function of expected goals conceded
  * league-wide assists per goal
  * yellow and red card rates per 90 by position
  * bonus points regressed on the events the model projects

Source data is the community season archive (vaastav/Fantasy-Premier-League),
which carries the per-gameweek detail the live API only exposes for the current
season.

    python scripts/fit_params.py                  # last completed season
    python scripts/fit_params.py --season 2025-26
    python scripts/fit_params.py --write          # patch the YAML in place
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import nbinom, poisson

from fplfh.config import CACHE_DIR, CONFIG_DIR

ARCHIVE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/"
    "data/{season}/gws/merged_gw.csv"
)
# Thresholds are FPL rules, not fitted quantities - see config/model_params.yaml
# for how they were verified against live gameweek data.
THRESHOLDS = {"DEF": 10, "MID": 12, "FWD": 12}


def load_season(season: str, refresh: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"history_{season}.csv"
    if refresh or not path.exists():
        url = ARCHIVE.format(season=season)
        print(f"downloading {url}")
        resp = requests.get(url, timeout=180)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    df = pd.read_csv(path)
    if "defensive_contribution" not in df.columns:
        raise SystemExit(
            f"season {season} has no defensive_contribution column - "
            "DefCon only exists from 2025-26 onward"
        )
    return df


def fit_defcon_dispersion(played: pd.DataFrame) -> dict[str, float]:
    """Solve for phi = var/mean reproducing the observed threshold hit-rate.

    Calibrating on the hit-rate rather than on the raw variance is deliberate:
    the model's only use of this distribution is P(X >= threshold), so that is
    the quantity worth getting right.
    """
    out: dict[str, float] = {}
    print("\n=== DefCon dispersion ===")
    for pos, thr in THRESHOLDS.items():
        sub = played[played["position"] == pos]
        rates = sub.groupby("element")["defensive_contribution"].agg(["count", "mean"])
        rates = rates[rates["count"] >= 8]
        if rates.empty:
            continue
        merged = sub.merge(rates[["mean"]], on="element")
        actual = float((merged["defensive_contribution"] >= thr).mean())

        def modelled(phi: float) -> float:
            mu = merged["mean"].to_numpy(dtype=float)
            if phi <= 1 + 1e-9:
                return float(poisson.sf(thr - 1, mu).mean())
            r = mu / (phi - 1.0)
            return float(nbinom.sf(thr - 1, r, r / (r + mu)).mean())

        try:
            phi = brentq(lambda p: modelled(p) - actual, 1.0001, 3.0)
        except ValueError:
            phi = 1.3
        out[pos] = round(float(phi), 3)
        pois = modelled(1.0)
        print(
            f"  {pos} thr>={thr}: observed {actual:.4f}  Poisson {pois:.4f}  "
            f"-> phi={phi:.3f}  (n={len(merged)})"
        )
    return out


def fit_saves(played: pd.DataFrame) -> dict[str, float]:
    gk = played[(played["position"] == "GK") & (played["expected_goals_conceded"] > 0)]
    x = gk["expected_goals_conceded"].to_numpy(dtype=float)
    y = gk["saves"].to_numpy(dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    corr = float(np.corrcoef(x, y)[0, 1])
    within = gk.groupby("element")["saves"].agg(["count", "mean", "var"])
    within = within[(within["count"] >= 10) & (within["mean"] > 0)]
    disp = float((within["var"] / within["mean"]).median())
    print("\n=== GK saves ===")
    print(f"  saves = {coef[0]:.4f} * xGC + {coef[1]:.4f}   (n={len(gk)}, r={corr:.3f})")
    print(f"  within-keeper var/mean = {disp:.3f}  (1.0 => Poisson)")
    return {"slope": round(float(coef[0]), 4), "intercept": round(float(coef[1]), 4),
            "dispersion": round(disp, 3)}


def fit_assists(df: pd.DataFrame) -> float:
    goals, assists = int(df["goals_scored"].sum()), int(df["assists"].sum())
    ratio = assists / goals if goals else 0.0
    print("\n=== Assists ===")
    print(f"  {assists} assists / {goals} goals = {ratio:.4f} per goal")
    return round(ratio, 4)


def fit_cards(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    print("\n=== Cards per 90 ===")
    yellow, red = {}, {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = df[(df["position"] == pos) & (df["minutes"] > 0)]
        n90 = sub["minutes"].sum() / 90.0
        key = "GKP" if pos == "GK" else pos
        yellow[key] = round(float(sub["yellow_cards"].sum() / n90), 4)
        red[key] = round(float(sub["red_cards"].sum() / n90), 4)
        print(f"  {key}: YC/90={yellow[key]:.4f}  RC/90={red[key]:.4f}")
    return {"yellow_per90": yellow, "red_per90": red}


def fit_bonus(played: pd.DataFrame) -> dict[str, dict[str, float]]:
    print("\n=== Bonus regression ===")
    df = played.copy()
    df["dc_hit"] = 0.0
    for pos, thr in THRESHOLDS.items():
        m = df["position"] == pos
        df.loc[m, "dc_hit"] = (df.loc[m, "defensive_contribution"] >= thr).astype(float)
    df["cs"] = (df["goals_conceded"] == 0).astype(float)

    feats = ["goals_scored", "assists", "cs", "dc_hit", "saves"]
    names = ["goals", "assists", "clean_sheet", "defcon_hit", "saves"]
    out: dict[str, dict[str, float]] = {}
    for pos in ["GK", "DEF", "MID", "FWD"]:
        sub = df[df["position"] == pos]
        if len(sub) < 50:
            continue
        X = np.hstack([sub[feats].to_numpy(dtype=float), np.ones((len(sub), 1))])
        y = sub["bonus"].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
        key = "GKP" if pos == "GK" else pos
        out[key] = {n: round(float(c), 3) for n, c in zip(names, coef[:-1])}
        out[key]["const"] = round(float(coef[-1]), 3)
        print(f"  {key}: R2={r2:.3f}  " +
              "  ".join(f"{n}={out[key][n]:+.3f}" for n in names))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--write", action="store_true", help="patch model_params.yaml")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_season(args.season, args.refresh)
    played = df[df["minutes"] >= 60]
    print(f"season {args.season}: {len(df)} player-gameweeks, {len(played)} with 60+ mins")

    results = {
        "defcon_dispersion": fit_defcon_dispersion(played),
        "saves": fit_saves(played),
        "assists_per_goal": fit_assists(df),
        "cards": fit_cards(df),
        "bonus": fit_bonus(played),
    }

    if args.write:
        import yaml
        path = CONFIG_DIR / "model_params.yaml"
        cfg = yaml.safe_load(io.open(path, encoding="utf-8"))
        cfg["defcon"]["dispersion"] = results["defcon_dispersion"]
        cfg["saves"].update(results["saves"])
        cfg["assists"]["per_goal"] = results["assists_per_goal"]
        cfg["cards"]["yellow_per90"] = results["cards"]["yellow_per90"]
        cfg["cards"]["red_per90"] = results["cards"]["red_per90"]
        cfg["bonus"]["coefficients"] = results["bonus"]
        with io.open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        print(f"\nWROTE {path}")
        print("NOTE: yaml.safe_dump drops the provenance comments. Check `git diff`,")
        print("      and restore the comment blocks if you intend to keep them.")
    else:
        print("\n(dry run - pass --write to patch config/model_params.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
