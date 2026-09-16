"""Configuration loading: model parameters, manual overrides, credentials."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
CACHE_DIR = ROOT / "data" / "cache"
OUT_DIR = ROOT / "data" / "out"

POSITIONS = ("GKP", "DEF", "MID", "FWD")
ELEMENT_TYPE_TO_POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


@lru_cache(maxsize=1)
def params() -> dict[str, Any]:
    """Model parameters (config/model_params.yaml)."""
    with open(CONFIG_DIR / "model_params.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def overrides() -> dict[str, Any]:
    """Manual per-player overrides. Re-read every call so notebook edits land."""
    path = CONFIG_DIR / "minutes_overrides.yaml"
    if not path.exists():
        return {"players": {}, "defaults": {}}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("players", {})
    data.setdefault("defaults", {})
    return data


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader - avoids a python-dotenv dependency.

    Existing environment variables win, so a shell export overrides the file.
    """
    path = path or (ROOT / ".env")
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def credentials() -> dict[str, str | None]:
    """API credentials from the environment. Never logged, never committed."""
    load_dotenv()
    return {
        "odds_api_key": os.getenv("ODDS_API_KEY"),
    }


def fpl_team_id() -> int | None:
    """Your FPL manager id, from the environment.

    Deliberately not part of ``credentials()``: a team id is public - anyone can
    fetch ``entry/<id>/`` - so it grants nothing and is not a secret. It lives in
    .env because it is *environment-specific config that should not be
    committed*, which is a different reason from secrecy. Hardcoding it in a
    tracked notebook would commit one person's setup to a shared repo.
    """
    load_dotenv()
    raw = os.getenv("FPL_TEAM_ID", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def ensure_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
