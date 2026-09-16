"""When will a flagged player be back?

FPL publishes a free-text ``news`` field for unavailable players. A minority of
those state a return date - "Expected back 11 Oct", "Suspended until 17 Oct" -
and where one is given it can be turned into a gameweek, so a multi-gameweek
projection can switch the player back on at the right point instead of writing
them off for the entire horizon.

**This covers a minority of cases.** Measured on a live squad, 193 players were
flagged and only about 13% carried a parseable date; the rest say "Unknown
return date". Those are frozen at today's availability, which is the honest
default: we genuinely do not know, and inventing a recovery curve would dress a
guess up as information.

Note the third category, which matters: some flagged players have *left the
league* ("Has joined Al Hilal permanently"). They are correctly zero forever, and
must never be restored by a recovery rule.
"""
from __future__ import annotations

import re
from datetime import date, datetime

import pandas as pd

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# "Expected back 11 Oct", "Suspended until 17 Oct", "back 7 Nov"
_DATE = re.compile(
    r"(?:back|until|returns?)\s+(?:on\s+)?(\d{1,2})\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)

# Departures - never restore these, whatever else the text says.
_GONE = re.compile(r"\b(joined|transferred|left|loan(?:ed)? to)\b", re.IGNORECASE)


def has_left(news: str | None) -> bool:
    """True when the player has moved club and will not return this season."""
    return bool(news) and bool(_GONE.search(str(news)))


def parse_return_date(news: str | None, today: date | None = None) -> date | None:
    """Extract a return date from FPL's news text, or None.

    The text carries a day and month but no year, so the year is inferred from
    the season's shape: a month at or after August belongs to the season's first
    calendar year, and January onward to the second.
    """
    if not news or has_left(news):
        return None
    m = _DATE.search(str(news))
    if not m:
        return None
    day, month = int(m.group(1)), _MONTHS[m.group(2).lower()]
    today = today or date.today()

    # A Premier League season runs August to May, so months Aug-Dec sit in the
    # earlier calendar year and Jan-May in the later one.
    year = today.year
    if month < 8 and today.month >= 8:
        year += 1
    elif month >= 8 and today.month < 8:
        year -= 1
    try:
        return date(year, month, day)
    except ValueError:
        return None   # e.g. "31 Feb" in malformed text


def return_gameweek(news: str | None, events: list[dict],
                    today: date | None = None) -> int | None:
    """The first gameweek a flagged player could feature in, or None.

    Returns the earliest gameweek whose deadline falls on or after the stated
    return date. None means either no date was given or the player has left.
    """
    when = parse_return_date(news, today)
    if when is None:
        return None
    for ev in sorted(events, key=lambda e: e["id"]):
        deadline = pd.to_datetime(ev["deadline_time"], utc=True).date()
        if deadline >= when:
            return int(ev["id"])
    return None


def return_gameweeks(players: pd.DataFrame, events: list[dict],
                     today: date | None = None) -> pd.Series:
    """Map player_id -> the gameweek they are expected back, where stated."""
    out = {}
    for _, r in players.iterrows():
        if r.get("status", "a") == "a":
            continue
        gw = return_gameweek(r.get("news", ""), events, today)
        if gw is not None:
            out[int(r["player_id"])] = gw
    return pd.Series(out, dtype="Int64", name="return_gw")
