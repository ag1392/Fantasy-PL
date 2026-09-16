"""Joining FPL entities to a third party's naming conventions.

Every odds provider names teams and players slightly differently, and the joins
are the most fragile part of any integration: an unmatched fixture silently
falls back to the model while the output still looks market-derived. So the
matching lives in one place, is shared by every provider, and always reports
what it failed to match rather than dropping it quietly.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# FPL names differ from most providers' in a handful of predictable ways.
TEAM_ALIASES = {
    "man city": {"manchester city", "man city"},
    "man utd": {"manchester utd", "manchester united", "man utd", "man united"},
    "spurs": {"tottenham", "tottenham hotspur", "spurs"},
    "nott'm forest": {"nottingham forest", "nottm forest", "notts forest", "forest"},
    "wolves": {"wolverhampton", "wolverhampton wanderers", "wolves"},
    "newcastle": {"newcastle utd", "newcastle united", "newcastle"},
    "brighton": {"brighton and hove albion", "brighton & hove albion", "brighton"},
    "west ham": {"west ham utd", "west ham united", "west ham"},
    "leeds": {"leeds utd", "leeds united", "leeds"},
    "leicester": {"leicester city", "leicester"},
    "ipswich town": {"ipswich", "ipswich town"},
    "hull city": {"hull", "hull city"},
    "coventry city": {"coventry", "coventry city"},
    "sheffield utd": {"sheffield united", "sheffield utd", "sheff utd"},
    "luton": {"luton town", "luton"},
    "bournemouth": {"afc bournemouth", "bournemouth"},
    "crystal palace": {"crystal palace", "palace"},
    "nottingham forest": {"nottingham forest", "nott'm forest"},
    "wolverhampton wanderers": {"wolves", "wolverhampton wanderers"},
}


# Letters that are atomic in Unicode rather than "base letter + accent", so
# NFKD leaves them intact and the ASCII filter would delete them. Without this
# "Odegaard" fails to match "Ødegaard", and worse, "Højlund" normalises to
# "h jlund" - the stripped letter becomes a space and splits the name in two.
_TRANSLITERATE = str.maketrans({
    "ø": "o", "Ø": "o", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "ł": "l", "Ł": "l", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe",
    "ß": "ss", "þ": "th", "Þ": "th", "ı": "i", "ʻ": "", "'": "", "’": "",
})


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation - for fuzzy name joins.

    Two passes are needed. Unicode NFKD decomposition separates diacritics from
    their base letter so the accent can be dropped (``Guéhi`` -> ``guehi``), but
    it does nothing for letters that are atomic in Unicode - those are
    transliterated explicitly first.
    """
    if text is None:
        return ""
    s = str(text).translate(_TRANSLITERATE)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# First tokens that identify a *city*, not a club, and so cannot be used as a
# shorthand key. "man" would otherwise make Man Utd and Man City interchangeable
# - and since both often kick off in the same window, the time filter would not
# catch the mismatch either.
AMBIGUOUS_STEMS = {
    "man", "west", "north", "south", "east", "sheffield", "bristol", "stoke",
    "swansea", "cardiff", "wigan", "bolton",
}
_MIN_STEM_LEN = 5


def team_key(name: str) -> set[str]:
    """Every spelling a team might legitimately appear under.

    The alias table is the primary mechanism. A bare first token is added only
    as a backstop, and only when it is long enough and unambiguous enough to
    identify one club on its own ("brighton" for "brighton and hove albion").
    """
    n = normalise(name)
    keys = {n}
    for canon, variants in TEAM_ALIASES.items():
        vs = {normalise(v) for v in variants} | {normalise(canon)}
        if n in vs:
            keys |= vs
    parts = n.split()
    if len(parts) > 1:
        stem = parts[0]
        if len(stem) >= _MIN_STEM_LEN and stem not in AMBIGUOUS_STEMS:
            keys.add(stem)
    return keys


def match_fixtures(
    fixtures_gw: pd.DataFrame,
    events: list[dict],
    home_key: str = "home_team",
    away_key: str = "away_team",
    time_key: str = "commence_time",
    id_key: str = "id",
    tolerance_hours: float = 30.0,
) -> tuple[dict[int, str], list[str]]:
    """Join FPL fixtures to a provider's event ids.

    Matched on team names **and** kick-off time. Names alone are ambiguous
    across competitions (a league fixture and a cup tie between the same two
    sides); time alone cannot separate simultaneous kick-offs.

    Returns ``(fixture_id -> event_id, unmatched_descriptions)``.
    """
    parsed = []
    for ev in events:
        parsed.append((
            ev.get(id_key),
            team_key(ev.get(home_key, "")),
            team_key(ev.get(away_key, "")),
            pd.to_datetime(ev.get(time_key), utc=True, errors="coerce"),
        ))

    mapping: dict[int, str] = {}
    unmatched: list[str] = []
    for _, fx in fixtures_gw.iterrows():
        h = team_key(fx.get("team_h_name", fx.get("team_h_short", "")))
        a = team_key(fx.get("team_a_name", fx.get("team_a_short", "")))
        ko = pd.to_datetime(fx["kickoff"], utc=True, errors="coerce")

        best = None
        for ev_id, eh, ea, start in parsed:
            if not (h & eh) or not (a & ea):
                continue
            gap = 0.0
            if pd.notna(ko) and pd.notna(start):
                gap = abs((start - ko).total_seconds()) / 3600.0
                if gap > tolerance_hours:
                    continue
            if best is None or gap < best[1]:
                best = (ev_id, gap)
        if best:
            mapping[int(fx["fixture_id"])] = str(best[0])
        else:
            unmatched.append(
                f"{fx.get('team_h_short', '?')} v {fx.get('team_a_short', '?')} "
                f"({fx.get('kickoff')})"
            )
    return mapping, unmatched
