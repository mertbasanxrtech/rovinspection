"""Central configuration.

Everything that can change inspection results is fixed here. Changing any of the
"result-affecting" values below requires bumping PROMPT_VERSION (or the model) so
the cache identity changes and the golden set is re-scored.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- Model (pinned stable ID; never a "latest" alias) -----------------------
GEMINI_MODEL = "gemini-3.8-flash"
THINKING_LEVEL = "LOW"              # lowest level gemini-3.8-flash supports (MINIMAL -> 400); pending PO approval
VOTES_PER_FRAME = 3                 # three independent responses per usable frame
REQUEST_TIMEOUT_S = 60
RETRY_ATTEMPTS = 5                  # transient errors only (429/500/503/504); exponential backoff
RETRY_INITIAL_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 60.0
RETRY_STATUS_CODES = (429, 500, 503, 504)

# --- Prompt -----------------------------------------------------------------
PROMPT_VERSION = "v0"               # bump on EVERY prompt/schema change (part of cache key)

# --- Frame extraction (result-affecting) ------------------------------------
FRAME_INTERVAL_S = 1.0              # one frame per second
FRAME_WIDTH = 1280                  # px, aspect ratio preserved
JPEG_QUALITY = 90                   # fixed encoder quality

# --- Labels -----------------------------------------------------------------
DETECTION_LABELS = ("structure", "marine_growth", "corrosion", "debris")
MARINE_GROWTH_LEVELS = ("low", "medium", "high")

# --- Budget (USD) -----------------------------------------------------------
PROJECT_BUDGET_USD = 200.0          # hard cap, whole project
DEMO_BUDGET_USD = 10.0              # hard cap, per processing run
PRICE_INPUT_PER_M = 0.75            # Gemini 3.8 Flash, Oct 2026 price list
PRICE_OUTPUT_PER_M = 3.75           # thinking tokens are billed as output

# --- Paths ------------------------------------------------------------------
DATA_DIR = ROOT / "data"
CLIPS_DIR = DATA_DIR / "clips"      # uploaded source videos
FRAMES_DIR = DATA_DIR / "frames"    # extracted 1 fps JPEGs
CACHE_DIR = DATA_DIR / "cache"      # Gemini response cache + spend ledger
GOLDEN_DIR = DATA_DIR / "golden"    # 40-frame answer key
OUTPUT_DIR = DATA_DIR / "output"    # annotated videos, findings, CSV
LOG_DIR = DATA_DIR / "logs"

ALL_DIRS = (CLIPS_DIR, FRAMES_DIR, CACHE_DIR, GOLDEN_DIR, OUTPUT_DIR, LOG_DIR)


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def gemini_api_key() -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return key or None


def token_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * PRICE_INPUT_PER_M + output_tokens * PRICE_OUTPUT_PER_M) / 1_000_000
