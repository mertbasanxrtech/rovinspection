"""Stage 1 tests on a synthetic video (10 fps, 5.5 s, 640x360, each frame a distinct grey level)."""
import hashlib
import json

import cv2
import numpy as np
import pytest

import config
import frames

FPS, N, W, H = 10, 55, 640, 360


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("src") / "test clip.mp4"
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert wr.isOpened()
    for i in range(N):
        wr.write(np.full((H, W, 3), i * 4, np.uint8))
    wr.release()
    return path


def run(video, root):
    return frames.extract(video, frames_root=root / "frames", clips_dir=root / "clips")


def test_one_frame_per_second(video, tmp_path):
    m = run(video, tmp_path)
    assert m["frame_count"] == 6                              # seconds 0..5
    assert [f["second"] for f in m["frames"]] == [0, 1, 2, 3, 4, 5]
    assert [f["source_frame_index"] for f in m["frames"]] == [0, 10, 20, 30, 40, 50]
    for f in m["frames"]:
        assert abs(f["source_ts_s"] - f["t_s"]) < 1.0 / FPS


def test_picks_correct_frame_content(video, tmp_path):
    m = run(video, tmp_path)
    d = tmp_path / "frames" / m["clip_id"]
    for f in m["frames"]:
        img = cv2.imread(str(d / f["file"]))
        assert abs(img.mean() - f["source_frame_index"] * 4) < 3


def test_width_and_aspect(video, tmp_path):
    m = run(video, tmp_path)
    assert m["frame_size"] == {"width": 1280, "height": 720}
    img = cv2.imread(str(tmp_path / "frames" / m["clip_id"] / m["frames"][0]["file"]))
    assert img.shape[:2] == (720, 1280)


def test_fingerprints_match_files(video, tmp_path):
    m = run(video, tmp_path)
    d = tmp_path / "frames" / m["clip_id"]
    for f in m["frames"]:
        assert hashlib.sha256((d / f["file"]).read_bytes()).hexdigest() == f["sha256"]


def test_two_runs_byte_identical(video, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ma, mb = run(video, a), run(video, b)
    run(video, a)                                             # re-run into same folder too
    da, db = a / "frames" / ma["clip_id"], b / "frames" / mb["clip_id"]
    files_a = sorted(p.name for p in da.iterdir())
    assert files_a == sorted(p.name for p in db.iterdir())
    for name in files_a:
        assert (da / name).read_bytes() == (db / name).read_bytes(), name


def test_clip_ingested_with_stable_id(video, tmp_path):
    m = run(video, tmp_path)
    assert m["clip_id"].startswith("test_clip_")
    assert (tmp_path / "clips" / f"{m['clip_id']}.mp4").is_file()
    saved = json.loads((tmp_path / "frames" / m["clip_id"] / "manifest.json").read_text())
    assert saved == m
    assert saved["settings"]["jpeg_quality"] == config.JPEG_QUALITY


def test_bad_file_raises(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(frames.FrameExtractionError):
        run(bad, tmp_path)
    with pytest.raises(frames.FrameExtractionError):
        run(tmp_path / "missing.mp4", tmp_path)
