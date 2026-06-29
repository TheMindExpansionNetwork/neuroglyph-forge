"""Situation-aware policy: thresholds, skill hints, and adaptation queue."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from neuroglyph_agent.policy import ActionRateLimiter, PredictionEvent, should_act

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE_PATH = ROOT / "data" / "adaptive_queue.json"


class Situation(str, Enum):
    DEV_PIPELINE = "dev_pipeline"
    COLLECT_EPOC = "collect_epoc"
    LIVE_BCI = "live_bci"
    TRAIN_FINETUNE = "train_finetune"
    INTEGRATION = "integration"
    CREATIVE_PROMO = "creative_promo"
    STREAM_NARRATIVE = "stream_narrative"
    OBSERVE_ONLY = "observe_only"


@dataclass(frozen=True)
class SituationProfile:
    confidence_min: float
    min_action_interval_sec: float
    allow_unreal: bool
    skill_hints: tuple[str, ...]
    doc_first: str


PROFILES: dict[Situation, SituationProfile] = {
    Situation.DEV_PIPELINE: SituationProfile(
        0.99, 999.0, False, ("neuroglyph-forge",), "docs/ROADMAP.md"
    ),
    Situation.COLLECT_EPOC: SituationProfile(
        0.99, 999.0, False, ("neuroglyph-forge",), "docs/CORTEX_SETUP.md"
    ),
    Situation.LIVE_BCI: SituationProfile(
        0.55, 0.5, True, ("neuroglyph-forge", "unreal-engine-mcp"), "docs/CORTEX_SETUP.md"
    ),
    Situation.TRAIN_FINETUNE: SituationProfile(
        0.99, 999.0, False, ("neuroglyph-forge",), "docs/MODAL_TRAINING.md"
    ),
    Situation.INTEGRATION: SituationProfile(
        0.99, 999.0, False, ("neuroglyph-forge", "hermes-agent"), "configs/hermes-mcp-snippet.yaml"
    ),
    Situation.CREATIVE_PROMO: SituationProfile(
        0.99, 999.0, False, ("comfyui",), "docs/COMFY_MCP.md"
    ),
    Situation.STREAM_NARRATIVE: SituationProfile(
        0.50, 0.25, True, ("neuroglyph-forge",), "docs/MINDBOT_INTEGRATION.md"
    ),
    Situation.OBSERVE_ONLY: SituationProfile(
        0.85, 1.0, False, ("neuroglyph-forge",), "docs/ADAPTIVE_ORCHESTRATION.md"
    ),
}


@dataclass
class SituationState:
    active: Situation = Situation.DEV_PIPELINE
    overlay: Situation | None = None  # e.g. OBSERVE_ONLY on top of LIVE_BCI

    def effective(self) -> Situation:
        return self.overlay or self.active

    def profile(self) -> SituationProfile:
        return PROFILES[self.effective()]


@dataclass
class QueuedGoal:
    id: str
    priority: int
    situation: str
    goal: str
    trigger: str = ""
    status: str = "pending"  # pending | active | paused | done


@dataclass
class AdaptiveQueue:
    version: int = 1
    active_situation: str = Situation.DEV_PIPELINE.value
    overlay: str | None = None
    items: list[QueuedGoal] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = DEFAULT_QUEUE_PATH) -> AdaptiveQueue:
        if not path.exists():
            return cls.default_seed()
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [QueuedGoal(**i) for i in data.get("items", [])]
        return cls(
            version=data.get("version", 1),
            active_situation=data.get("active_situation", Situation.DEV_PIPELINE.value),
            overlay=data.get("overlay"),
            items=items,
        )

    @classmethod
    def default_seed(cls) -> AdaptiveQueue:
        return cls(
            items=[
                QueuedGoal(
                    id="goal-real-hand-60",
                    priority=10,
                    situation=Situation.COLLECT_EPOC.value,
                    goal="Hand classifier >60% validation accuracy on real EPOC typing sessions",
                    trigger="cortex_credentials_set",
                ),
                QueuedGoal(
                    id="goal-modal-finetune",
                    priority=8,
                    situation=Situation.TRAIN_FINETUNE.value,
                    goal="Run Modal GPU fine-tune after first real processed_hand.pt upload",
                    trigger="real_processed_hand_exists",
                ),
                QueuedGoal(
                    id="goal-mindbot-bus",
                    priority=7,
                    situation=Situation.STREAM_NARRATIVE.value,
                    goal="Emit mindbot_step_v1 JSONL on every gated BCI action in live demo",
                    trigger="live_bci_demo_stable",
                ),
                QueuedGoal(
                    id="goal-adaptive-doc",
                    priority=5,
                    situation=Situation.DEV_PIPELINE.value,
                    goal="Keep ADAPTIVE_ORCHESTRATION.md and queue in sync with shipped code",
                    trigger="phase_4_mindbot",
                    status="done",
                ),
            ]
        )

    def save(self, path: Path = DEFAULT_QUEUE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "active_situation": self.active_situation,
            "overlay": self.overlay,
            "items": [g.__dict__ for g in self.items],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def state(self) -> SituationState:
        active = Situation(self.active_situation)
        overlay = Situation(self.overlay) if self.overlay else None
        return SituationState(active=active, overlay=overlay)


class SituationRouter:
    """Applies situation-specific gates to prediction events."""

    def __init__(self, state: SituationState | None = None):
        self.state = state or SituationState()
        prof = self.state.profile()
        self._limiter = ActionRateLimiter(min_interval_sec=prof.min_action_interval_sec)
        self._conf_min = prof.confidence_min

    def should_emit(self, event: PredictionEvent) -> bool:
        prof = self.state.profile()
        if not prof.allow_unreal and event.mode == "unreal_control":
            return False
        if event.confidence < self._conf_min:
            return False
        if not should_act(event):
            return False
        return self._limiter.should_emit(event)

    def hermes_context_block(self) -> dict[str, Any]:
        prof = self.state.profile()
        return {
            "situation": self.state.effective().value,
            "confidence_min": prof.confidence_min,
            "skill_hints": list(prof.skill_hints),
            "doc_first": prof.doc_first,
            "allow_unreal": prof.allow_unreal,
        }


def infer_situation_from_env(*, cortex_mock: bool = True, checkpoint_val_acc: float | None = None) -> SituationState:
    """Heuristic bootstrap when user has not set queue manually."""
    if not cortex_mock:
        base = Situation.LIVE_BCI
    else:
        base = Situation.DEV_PIPELINE
    overlay = None
    if checkpoint_val_acc is not None and checkpoint_val_acc < 0.6:
        overlay = Situation.OBSERVE_ONLY
    return SituationState(active=base, overlay=overlay)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Adaptive situation + queue")
    p.add_argument("--show", action="store_true")
    p.add_argument("--set", type=str, choices=[s.value for s in Situation])
    p.add_argument("--overlay", type=str, choices=[s.value for s in Situation])
    p.add_argument("--clear-overlay", action="store_true")
    p.add_argument("--seed", action="store_true", help="Write default queue file")
    args = p.parse_args(argv)

    q = AdaptiveQueue.load()
    if args.seed or not DEFAULT_QUEUE_PATH.exists():
        q = AdaptiveQueue.default_seed()
        q.save()
        print(f"seeded {DEFAULT_QUEUE_PATH}")

    if args.set:
        q.active_situation = args.set
        q.save()
    if args.overlay:
        q.overlay = args.overlay
        q.save()
    if args.clear_overlay:
        q.overlay = None
        q.save()

    if args.show or not any([args.set, args.overlay, args.clear_overlay, args.seed]):
        print(json.dumps(q.__dict__, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o), indent=2))
        router = SituationRouter(q.state())
        print("hermes_block:", json.dumps(router.hermes_context_block(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())