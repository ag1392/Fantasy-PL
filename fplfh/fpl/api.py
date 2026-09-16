"""Thin, cached client for the public Fantasy Premier League API.

No authentication needed for any endpoint used here. Responses are cached on
disk so a modelling loop does not hammer the API; pass ``max_age=0`` to force
a refresh (do this after a price change or a team-news update).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from ..config import CACHE_DIR

BASE = "https://fantasy.premierleague.com/api"
UA = "Mozilla/5.0 (compatible; Fantasy-PL/0.1; +local research tool)"
DEFAULT_MAX_AGE = 3600.0  # seconds


class FPLClient:
    """Cached reader for FPL endpoints."""

    def __init__(self, cache_dir: Path | None = None, max_age: float = DEFAULT_MAX_AGE):
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age = max_age
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": UA})

    # ------------------------------------------------------------------ core
    def _get(self, path: str, cache_key: str, max_age: float | None = None) -> Any:
        max_age = self.max_age if max_age is None else max_age
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists() and max_age > 0:
            if time.time() - cache_file.stat().st_mtime < max_age:
                with open(cache_file, encoding="utf-8") as fh:
                    return json.load(fh)
        resp = self.session.get(f"{BASE}/{path}", timeout=40)
        resp.raise_for_status()
        data = resp.json()
        # utf-8 explicitly: player names contain non-cp1252 characters and the
        # Windows default codec silently corrupts them.
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        return data

    # ------------------------------------------------------------- endpoints
    def bootstrap(self, max_age: float | None = None) -> dict:
        """Players, teams, events, scoring rules, game settings."""
        return self._get("bootstrap-static/", "bootstrap", max_age)

    def fixtures(self, max_age: float | None = None) -> list[dict]:
        """All fixtures for the season."""
        return self._get("fixtures/", "fixtures", max_age)

    def live(self, event: int, max_age: float | None = None) -> dict:
        """Per-player stats and the points `explain` breakdown for a gameweek."""
        return self._get(f"event/{event}/live/", f"live_{event}", max_age)

    def element_summary(self, element_id: int, max_age: float | None = None) -> dict:
        """One player's fixture-by-fixture history."""
        return self._get(
            f"element-summary/{element_id}/", f"element_{element_id}", max_age
        )

    def entry(self, entry_id: int, max_age: float | None = None) -> dict:
        """A manager's profile: name, overall rank, season summary."""
        return self._get(f"entry/{entry_id}/", f"entry_{entry_id}", max_age)

    def entry_picks(self, entry_id: int, event: int, max_age: float | None = None) -> dict:
        """A manager's 15 picks for one gameweek, with bank and squad value.

        Public, but only once a gameweek's deadline has passed - picks for an
        upcoming gameweek are private until then. ``latest_picks`` handles that.
        """
        return self._get(
            f"entry/{entry_id}/event/{event}/picks/", f"picks_{entry_id}_{event}", max_age
        )

    def latest_picks(self, entry_id: int, max_age: float | None = None) -> tuple[dict, int]:
        """The most recent squad the API will show, and which gameweek it is.

        Walks back from the current gameweek until a gameweek returns picks,
        since the upcoming one stays private until its deadline passes.
        """
        current = self.current_event() or 1
        last_error: Exception | None = None
        for ev in range(current, 0, -1):
            try:
                return self.entry_picks(entry_id, ev, max_age), ev
            except Exception as exc:  # noqa: BLE001 - try the previous gameweek
                last_error = exc
        raise RuntimeError(
            f"no picks available for entry {entry_id} - check the ID is right "
            f"(last error: {last_error})"
        )

    # ---------------------------------------------------------------- helpers
    def current_event(self) -> int | None:
        for ev in self.bootstrap()["events"]:
            if ev.get("is_current"):
                return ev["id"]
        return None

    def next_event(self) -> int | None:
        for ev in self.bootstrap()["events"]:
            if ev.get("is_next"):
                return ev["id"]
        return None

    def target_event(self) -> int:
        """The gameweek a Free Hit would most likely be played on: the next one
        still open for transfers, falling back to the current one."""
        return self.next_event() or self.current_event() or 1

    def finished_events(self) -> list[int]:
        return [
            ev["id"]
            for ev in self.bootstrap()["events"]
            if ev.get("finished") and ev.get("data_checked")
        ]
