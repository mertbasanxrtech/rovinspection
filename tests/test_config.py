import config


def test_model_is_pinned():
    assert config.GEMINI_MODEL == "gemini-3.8-flash"
    assert "latest" not in config.GEMINI_MODEL


def test_fixed_inputs():
    assert config.FRAME_INTERVAL_S == 1.0
    assert config.FRAME_WIDTH == 1280
    assert 1 <= config.JPEG_QUALITY <= 100
    assert config.VOTES_PER_FRAME == 3
    assert config.THINKING_LEVEL == "LOW"  # MINIMAL unsupported on gemini-3.8-flash
    assert config.PROMPT_VERSION


def test_labels():
    assert config.DETECTION_LABELS == ("structure", "marine_growth", "corrosion", "debris")
    assert config.MARINE_GROWTH_LEVELS == ("low", "medium", "high")


def test_budget_defaults():
    assert config.PROJECT_BUDGET_USD == 200.0
    assert config.DEMO_BUDGET_USD == 10.0


def test_cost_formula():
    # 1M input + 1M output = $0.75 + $3.75
    assert abs(config.token_cost_usd(1_000_000, 1_000_000) - 4.50) < 1e-9
    # build-plan estimate: 2.7M in + 0.54M out ≈ $4.05
    assert abs(config.token_cost_usd(2_700_000, 540_000) - 4.05) < 1e-9


def test_ensure_dirs():
    config.ensure_dirs()
    assert all(d.is_dir() for d in config.ALL_DIRS)
