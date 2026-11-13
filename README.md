# ROV Inspection AI — Phase 1 Prototype

## Setup (Stage 0)

1. Install Python 3.10+ and FFmpeg (`ffmpeg` and `ffprobe` must be on PATH).
   - Windows: `winget install Gyan.FFmpeg`
   - macOS: `brew install ffmpeg`
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (Google AI Studio, billing enabled).
4. Run the checks:
   ```
   python check_setup.py
   pytest -q
   ```

## Layout

| File | Stage | Purpose |
|---|---|---|
| `config.py` | 0 | All fixed settings: model, prompt version, frame rate, size, JPEG quality, budgets, prices, paths |
| `check_setup.py` | 0 | Environment + Gemini connection check |
| `frames.py` | 1 | Frame extraction (1 fps, 1280 px, timestamp, fingerprint) |
| `clean.py` | 2 | Dark / blur detection → "Not Inspectable" |
| `golden.py` | 3 | Golden-set selection and labelling helper |
| `analyse.py` | 4 | Gemini calls, JSON schema, cache, cost ledger, spend stop |
| `findings.py` | 5 | Majority vote, neighbouring-frame rule, merging |
| `evaluate.py` | 6 | Accuracy + run-to-run agreement on golden set |
| `draw.py` | 7 | Annotated video |
| `app.py` | 8 | Streamlit demo |
| `data/` | — | clips, frames, cache, golden, output, logs (git-ignored) |
