"""Stage 1 — Frame extraction.

Usage:
    python frames.py <video> [<video> ...]

For each video:
  1. Ingest: copy into data/clips/<clip_id><ext>  (clip_id = <name>_<first 12 hex of file SHA-256>)
  2. Extract one frame per second (first decoded frame at/after each whole second)
  3. Resize to FRAME_WIDTH px wide (aspect kept), encode JPEG at JPEG_QUALITY
  4. Save data/frames/<clip_id>/frame_<second>.jpg + manifest.json + frames.csv

Re-running on the same video on the same machine produces byte-identical output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

import config

FRAME_TOLERANCE_S = 0.001  # float slack when comparing frame timestamps to the target second


class FrameExtractionError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_clip_id(path: Path, sha: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_")[:40] or "clip"
    return f"{stem}_{sha[:12]}"


def probe(path: Path) -> dict:
    """Video metadata via ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,avg_frame_rate,nb_frames,codec_name:format=duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise FrameExtractionError(f"ffprobe failed on {path}: {e}") from e
    data = json.loads(out)
    if not data.get("streams"):
        raise FrameExtractionError(f"No video stream in {path}")
    s = data["streams"][0]
    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {
        "codec": s.get("codec_name"),
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": round(fps, 3),
        "duration_s": round(float(data.get("format", {}).get("duration") or 0.0), 3),
    }


def resize_to_width(img, width: int = config.FRAME_WIDTH):
    h, w = img.shape[:2]
    if w == width:
        return img
    new_h = int(round(h * width / w))
    interp = cv2.INTER_AREA if w > width else cv2.INTER_CUBIC
    return cv2.resize(img, (width, new_h), interpolation=interp)


def encode_jpeg(img) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    if not ok:
        raise FrameExtractionError("JPEG encoding failed")
    return buf.tobytes()


def ingest(video: Path, clips_dir: Path) -> tuple[str, str, Path]:
    if not video.is_file():
        raise FrameExtractionError(f"File not found: {video}")
    sha = file_sha256(video)
    clip_id = make_clip_id(video, sha)
    clips_dir.mkdir(parents=True, exist_ok=True)
    dest = clips_dir / f"{clip_id}{video.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(video, dest)
    return clip_id, sha, dest


def extract(video: str | Path, frames_root: Path = config.FRAMES_DIR,
            clips_dir: Path = config.CLIPS_DIR) -> dict:
    video = Path(video)
    clip_id, sha, stored = ingest(video, clips_dir)
    meta = probe(stored)

    cap = cv2.VideoCapture(str(stored))
    if not cap.isOpened():
        raise FrameExtractionError(f"OpenCV cannot open {stored}")

    out_dir = frames_root / clip_id
    if out_dir.exists():
        shutil.rmtree(out_dir)          # always rebuild from scratch → no stale files
    out_dir.mkdir(parents=True)

    frames, idx, next_t = [], -1, 0.0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            idx += 1
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts + FRAME_TOLERANCE_S < next_t:
                continue
            second = int(round(next_t / config.FRAME_INTERVAL_S))
            jpg = encode_jpeg(resize_to_width(img))
            name = f"frame_{second:06d}.jpg"
            (out_dir / name).write_bytes(jpg)
            frames.append({
                "second": second,
                "t_s": round(next_t, 3),
                "source_ts_s": round(ts, 3),
                "source_frame_index": idx,
                "file": name,
                "sha256": hashlib.sha256(jpg).hexdigest(),
            })
            # skip ahead past any targets this frame already covers
            while next_t <= ts + FRAME_TOLERANCE_S:
                next_t = round(next_t + config.FRAME_INTERVAL_S, 6)
    finally:
        cap.release()

    if not frames:
        raise FrameExtractionError(f"No frames decoded from {stored}")

    sample = cv2.imread(str(out_dir / frames[0]["file"]))
    manifest = {
        "clip_id": clip_id,
        "source_file": video.name,
        "source_sha256": sha,
        "source": meta,
        "settings": {
            "frame_interval_s": config.FRAME_INTERVAL_S,
            "frame_width": config.FRAME_WIDTH,
            "jpeg_quality": config.JPEG_QUALITY,
            "opencv_version": cv2.__version__,
        },
        "frame_size": {"width": int(sample.shape[1]), "height": int(sample.shape[0])},
        "frame_count": len(frames),
        "frames": frames,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                           encoding="utf-8")
    with open(out_dir / "frames.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(frames[0].keys()))
        wr.writeheader()
        wr.writerows(frames)
    return manifest


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    config.ensure_dirs()
    rc = 0
    for v in argv:
        try:
            m = extract(v)
            s = m["source"]
            print(f"[OK] {v}\n     clip_id={m['clip_id']}  source={s['width']}x{s['height']} "
                  f"@{s['fps']}fps  duration={s['duration_s']}s\n"
                  f"     frames={m['frame_count']}  size={m['frame_size']['width']}x"
                  f"{m['frame_size']['height']}\n     output={config.FRAMES_DIR / m['clip_id']}")
        except FrameExtractionError as e:
            print(f"[FAIL] {v}: {e}")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))