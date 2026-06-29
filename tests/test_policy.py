def test_rate_limiter_blocks_burst():
    from neuroglyph_agent.policy import ActionRateLimiter, PredictionEvent

    lim = ActionRateLimiter(min_interval_sec=10.0)
    ev = PredictionEvent("t", "left", 0.9, "unreal_control", 1)
    assert lim.should_emit(ev) is True
    assert lim.should_emit(ev) is False