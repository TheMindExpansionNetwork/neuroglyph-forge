def test_situation_observe_blocks_low_confidence():
    from neuroglyph_agent.policy import PredictionEvent
    from neuroglyph_agent.situations import Situation, SituationRouter, SituationState

    router = SituationRouter(SituationState(active=Situation.OBSERVE_ONLY))
    ev = PredictionEvent("t", "left", 0.7, "unreal_control", 1)
    assert router.should_emit(ev) is False
    ev2 = PredictionEvent("t", "left", 0.9, "typing", 2)
    assert router.should_emit(ev2) is True


def test_adaptive_queue_seed():
    from neuroglyph_agent.situations import AdaptiveQueue

    q = AdaptiveQueue.default_seed()
    assert any(g.id == "goal-real-hand-60" for g in q.items)