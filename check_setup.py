"""Stage 0 environment check.

Usage:
    python check_setup.py               # full check incl. one tiny Gemini call (~$0.0001)
    python check_setup.py --skip-gemini # local checks only
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import config  # noqa: E402

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def check_python() -> None:
    v = sys.version_info
    record("Python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}")


def check_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        if not path:
            record(f"{tool} on PATH", False, "not found")
            continue
        out = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout
        record(f"{tool} on PATH", True, out.splitlines()[0] if out else path)


def check_packages() -> None:
    for mod, label in (("cv2", "opencv"), ("numpy", "numpy"), ("google.genai", "google-genai"),
                       ("streamlit", "streamlit"), ("dotenv", "python-dotenv")):
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            if mod == "google.genai":
                from google.genai import version as gv
                ver = gv.__version__
            record(f"import {label}", True, ver)
        except Exception as e:  # noqa: BLE001
            record(f"import {label}", False, repr(e))
    try:
        from google.genai import types
        ok = hasattr(types, "ThinkingLevel") and hasattr(types.ThinkingLevel, config.THINKING_LEVEL)
        record(f"SDK knows thinking_level={config.THINKING_LEVEL} (model checked by test call)", ok,
               "" if ok else "upgrade: pip install -U google-genai")
    except Exception as e:  # noqa: BLE001
        record(f"SDK knows thinking_level={config.THINKING_LEVEL}", False, repr(e))


def check_dirs() -> None:
    config.ensure_dirs()
    missing = [str(d) for d in config.ALL_DIRS if not d.is_dir()]
    record("data/ folders", not missing, ", ".join(missing))


def check_key() -> bool:
    key = config.gemini_api_key()
    return record("GEMINI_API_KEY in .env", bool(key),
                  f"found ({len(key)} chars)" if key else "copy .env.example to .env and set it")


def check_gemini() -> None:
    from google import genai
    from google.genai import types

    retry = types.HttpRetryOptions(attempts=config.RETRY_ATTEMPTS,
                                   initial_delay=config.RETRY_INITIAL_DELAY_S,
                                   max_delay=config.RETRY_MAX_DELAY_S,
                                   http_status_codes=list(config.RETRY_STATUS_CODES))
    client = genai.Client(api_key=config.gemini_api_key(),
                          http_options=types.HttpOptions(timeout=config.REQUEST_TIMEOUT_S * 1000,
                                                         retry_options=retry))
    # 1) pinned model ID exists
    try:
        m = client.models.get(model=config.GEMINI_MODEL)
        record(f"model '{config.GEMINI_MODEL}' available", True, getattr(m, "name", ""))
    except Exception as e:  # noqa: BLE001
        record(f"model '{config.GEMINI_MODEL}' available", False, repr(e))
        return
    # 2) one tiny call with the planned thinking setting
    try:
        resp = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents="Reply with exactly: OK",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=config.THINKING_LEVEL)),
        )
        u = resp.usage_metadata
        inp = u.prompt_token_count or 0
        out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        record("Gemini test call", bool(resp.text), f"reply={resp.text!r}")
        print(f"       tokens: input={inp} output={u.candidates_token_count} "
              f"thinking={u.thoughts_token_count} cost≈${config.token_cost_usd(inp, out):.6f}")
    except Exception as e:  # noqa: BLE001
        record("Gemini test call", False, repr(e))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gemini", action="store_true")
    args = ap.parse_args()

    check_python()
    check_ffmpeg()
    check_packages()
    check_dirs()
    has_key = check_key()
    if args.skip_gemini:
        print("[SKIP] Gemini checks (--skip-gemini)")
    elif has_key:
        check_gemini()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
