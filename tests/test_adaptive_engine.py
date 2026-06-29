def test_trigger_cortex():
    from neuroglyph_agent.adaptive_engine import WorldSignals, trigger_satisfied

    assert trigger_satisfied("cortex_credentials_set", WorldSignals(cortex_credentials=True))
    assert not trigger_satisfied("cortex_credentials_set", WorldSignals())


def test_reconcile_marks_mindbot_done():
    from neuroglyph_agent.adaptive_engine import WorldSignals, reconcile_queue
    from neuroglyph_agent.situations import AdaptiveQueue, QueuedGoal

    q = AdaptiveQueue(items=[
        QueuedGoal(
            id="goal-mindbot-bus",
            priority=7,
            situation="stream_narrative",
            goal="bus",
            trigger="live_bci_demo_stable",
            status="pending",
        )
    ])
    sig = WorldSignals(mindbot_live_steps=5)
    reconcile_queue(q, sig)
    assert q.items[0].status == "done"


def test_brief_has_commands():
    from neuroglyph_agent.adaptive_engine import build_brief
    from neuroglyph_agent.situations import AdaptiveQueue

    brief = build_brief(AdaptiveQueue.default_seed())
    assert brief.hermes_block["situation"]
    assert isinstance(brief.next_commands, list)


def test_propose_collect_when_creds_no_raw():
    from neuroglyph_agent.adaptive_engine import collect_world_signals, propose_transitions
    from neuroglyph_agent.situations import AdaptiveQueue, Situation

    sig = collect_world_signals()
    sig.cortex_credentials = True
    sig.raw_session_dirs = 0
    q = AdaptiveQueue(active_situation=Situation.DEV_PIPELINE.value)
    props = propose_transitions(sig, q)
    assert any(p.to_situation == Situation.COLLECT_EPOC.value for p in props)