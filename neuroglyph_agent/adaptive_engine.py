"""
Deeper adaptive layer: world signals, trigger evaluation, transitions, agent brief.

Synergetic loop:
  perceive (signals) → infer situation → reconcile queue → recommend actions → log narrative
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuroglyph_agent.situations import (
    DEFAULT_QUEUE_PATH,
    AdaptiveQueue,
    QueuedGoal,
    Situation,
    SituationRouter,
    SituationState,
)

ROOT = Path(__file__).resolve().parents[1]
BRIEF_PATH = ROOT / "data" / "adaptive_brief.md"
HISTORY_PATH = ROOT / "data" / "adaptive_history.jsonl"


@dataclass
class WorldSignals:
    """Observable project + environment facts (no hallucination)."""

    cortex_credentials: bool = False
    raw_session_dirs: int = 0
    raw_eeg_rows_min: int = 0
    has_real_processed_hand: bool = False
    synthetic_only_data: bool = True
    checkpoint_exists: bool = False
    checkpoint_val_acc: float | None = None
    mindbot_live_steps: int = 0
    pytest_last_ok: bool | None = None
    modal_cli: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dotenv_keys() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in (ROOT / ".env", Path(os.environ.get("USERPROFILE", "")) / "AppData/Local/hermes/.env"):
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def collect_world_signals(root: Path = ROOT) -> WorldSignals:
    env = {**_load_dotenv_keys(), **os.environ}
    sig = WorldSignals()
    sig.cortex_credentials = bool(
        env.get("EMOTIV_CLIENT_ID") and env.get("EMOTIV_CLIENT_SECRET")
    )

    raw = root / "data" / "raw"
    if raw.exists():
        dirs = [d for d in raw.iterdir() if d.is_dir() and d.name.startswith("session_")]
        sig.raw_session_dirs = len(dirs)
        rows_min = 10**9
        for d in dirs[:5]:
            eeg = d / "eeg.csv"
            if eeg.exists():
                rows_min = min(rows_min, max(0, sum(1 for _ in eeg.open()) - 1))
        sig.raw_eeg_rows_min = rows_min if rows_min < 10**9 else 0

    proc_meta = root / "data" / "processed" / "session_meta.json"
    if proc_meta.exists():
        meta = json.loads(proc_meta.read_text(encoding="utf-8"))
        sid = str(meta.get("session_id", ""))
        sig.synthetic_only_data = sid.startswith("synthetic") or meta.get("synthetic", False)
        sig.has_real_processed_hand = (root / "data" / "processed" / "processed_hand.pt").exists() and not sig.synthetic_only_data

    ckpt = root / "checkpoints" / "tiny_b2q_hand.pt"
    sig.checkpoint_exists = ckpt.exists()
    if ckpt.exists():
        try:
            import torch

            blob = torch.load(ckpt, weights_only=True)
            sig.checkpoint_val_acc = float(blob.get("val_acc", 0))
        except Exception:
            sig.checkpoint_val_acc = None

    bus = root / "data" / "mindbot_export" / "live_steps.jsonl"
    if bus.exists():
        sig.mindbot_live_steps = sum(1 for _ in bus.open(encoding="utf-8"))

    marker = root / "tests" / ".last_pytest_ok"
    if marker.exists():
        sig.pytest_last_ok = marker.read_text().strip() == "1"

    import shutil

    sig.modal_cli = shutil.which("modal") is not None
    return sig


TRIGGER_FUNCS: dict[str, Any] = {}


def _trigger(name: str):
    def deco(fn):
        TRIGGER_FUNCS[name] = fn
        return fn

    return deco


@_trigger("cortex_credentials_set")
def _t_cortex(s: WorldSignals) -> bool:
    return s.cortex_credentials


@_trigger("real_processed_hand_exists")
def _t_real_proc(s: WorldSignals) -> bool:
    return s.has_real_processed_hand


@_trigger("live_bci_demo_stable")
def _t_live_bus(s: WorldSignals) -> bool:
    return s.mindbot_live_steps >= 1


@_trigger("phase_4_mindbot")
def _t_always(_: WorldSignals) -> bool:
    return True


@_trigger("always_pending")
def _t_never(_: WorldSignals) -> bool:
    return False


def trigger_satisfied(name: str, signals: WorldSignals) -> bool:
    if not name:
        return True
    fn = TRIGGER_FUNCS.get(name)
    if fn is None:
        return False
    return bool(fn(signals))


@dataclass
class TransitionProposal:
    from_situation: str
    to_situation: str
    overlay: str | None
    reason: str
    confidence: float  # 0-1 how strong the recommendation is


def propose_transitions(signals: WorldSignals, queue: AdaptiveQueue) -> list[TransitionProposal]:
    """Rule-based situation inference (transparent, editable)."""
    props: list[TransitionProposal] = []
    active = queue.active_situation

    if signals.raw_session_dirs >= 1 and not signals.has_real_processed_hand:
        props.append(
            TransitionProposal(
                active,
                Situation.TRAIN_FINETUNE.value,
                None,
                "Raw EPOC sessions exist but no real processed_hand — run make_epochs_cli",
                0.85,
            )
        )

    if signals.cortex_credentials and signals.raw_session_dirs == 0:
        props.append(
            TransitionProposal(
                active,
                Situation.COLLECT_EPOC.value,
                None,
                "Cortex creds set, no raw sessions yet",
                0.9,
            )
        )

    if signals.cortex_credentials and signals.has_real_processed_hand:
        props.append(
            TransitionProposal(
                active,
                Situation.LIVE_BCI.value,
                None,
                "Real data pipeline ready — live demo path",
                0.8,
            )
        )

    overlay = None
    if signals.has_real_processed_hand and signals.checkpoint_val_acc is not None:
        if signals.checkpoint_val_acc < 0.6:
            overlay = Situation.OBSERVE_ONLY.value
            props.append(
                TransitionProposal(
                    active,
                    active,
                    overlay,
                    f"Real-data checkpoint val_acc={signals.checkpoint_val_acc:.2f} < 0.6",
                    0.95,
                )
            )
    elif signals.synthetic_only_data and signals.checkpoint_val_acc == 1.0:
        if active in (Situation.LIVE_BCI.value, Situation.STREAM_NARRATIVE.value):
            props.append(
                TransitionProposal(
                    active,
                    active,
                    Situation.OBSERVE_ONLY.value,
                    "Synthetic-perfect checkpoint — Unreal/typing actuation is smoke only",
                    0.75,
                )
            )

    if not props and active == Situation.DEV_PIPELINE.value and signals.pytest_last_ok:
        props.append(
            TransitionProposal(
                active,
                Situation.TRAIN_FINETUNE.value,
                None,
                "Tests green, no hardware — dataset/Modal work is highest leverage",
                0.55,
            )
        )

    return props


GOAL_PLAYBOOK: dict[str, list[str]] = {
    "goal-real-hand-60": [
        "docs/CORTEX_SETUP.md — set EMOTIV_CLIENT_ID/SECRET in .env",
        "python scripts/collect_session.py --duration 600  # add --live when ready",
        "python -m neuroglyph_data.make_epochs_cli --raw data/raw --out data/processed --task hand",
        "python -m neuroglyph_train.finetune_subject --base-checkpoint checkpoints/tiny_b2q_hand.pt",
        "python -m neuroglyph_train.evaluate --checkpoint checkpoints/tiny_b2q_hand.pt",
    ],
    "goal-modal-finetune": [
        "pip install -e '.[cloud]' && modal setup",
        "modal volume put neuroglyph-data data/processed /processed",
        "modal run neuroglyph_cloud/modal_train.py --finetune --epochs 30",
    ],
    "goal-mindbot-bus": [
        "python scripts/live_bci_demo.py --situation stream_narrative --seconds 30",
        "wc -l data/mindbot_export/live_steps.jsonl",
    ],
    "goal-unreal-integration": [
        "docs/UNREAL_SETUP.md",
        "hermes mcp test unreal-mcp",
        "python -c \"from neuroglyph_agent.mcp_server import call_tool; print(call_tool('send_prediction_to_unreal', {'prediction':'left','confidence':0.9}))\"",
    ],
}


def completion_satisfied(goal_id: str, signals: WorldSignals) -> bool:
    if goal_id == "goal-real-hand-60":
        return (
            signals.has_real_processed_hand
            and signals.checkpoint_val_acc is not None
            and signals.checkpoint_val_acc >= 0.6
        )
    if goal_id == "goal-modal-finetune":
        return signals.has_real_processed_hand and signals.modal_cli
    if goal_id == "goal-mindbot-bus":
        return signals.mindbot_live_steps >= 3
    if goal_id == "goal-adaptive-doc":
        return True
    return False


@dataclass
class AgentBrief:
    generated_at: float
    signals: WorldSignals
    queue: AdaptiveQueue
    transitions: list[TransitionProposal]
    active_goal: QueuedGoal | None
    next_commands: list[str]
    hermes_block: dict[str, Any]
    narrative: str

    def to_markdown(self) -> str:
        lines = [
            "# NeuroGlyph adaptive brief",
            f"_generated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.generated_at))}_",
            "",
            "## Situation",
            f"- **active:** `{self.queue.active_situation}`",
            f"- **overlay:** `{self.queue.overlay}`",
            f"- **effective:** `{self.hermes_block.get('situation')}`",
            "",
            "## World signals",
            "```json",
            json.dumps(self.signals.to_dict(), indent=2),
            "```",
            "",
            "## Top transition proposals",
        ]
        for t in self.transitions[:3]:
            lines.append(f"- **{t.reason}** → `{t.to_situation}` overlay=`{t.overlay}` ({t.confidence:.0%})")
        lines.extend(["", "## Active queue goal"])
        if self.active_goal:
            lines.append(f"- **{self.active_goal.id}** (P{self.active_goal.priority}): {self.active_goal.goal}")
        else:
            lines.append("- _(none pending for current situation)_")
        lines.extend(["", "## Suggested commands (agent: run highest feasible)", ""])
        for cmd in self.next_commands[:8]:
            lines.append(f"- `{cmd}`")
        lines.extend(["", "## Narrative (MindBot / stream)", "", self.narrative, ""])
        return "\n".join(lines)


def reconcile_queue(queue: AdaptiveQueue, signals: WorldSignals) -> AdaptiveQueue:
    """Update goal statuses from triggers and completions."""
    for g in queue.items:
        if g.status == "done":
            continue
        if completion_satisfied(g.id, signals):
            g.status = "done"
            continue
        if trigger_satisfied(g.trigger, signals) and g.status == "pending":
            g.status = "active"
    return queue


def pick_active_goal(queue: AdaptiveQueue) -> QueuedGoal | None:
    pending = [
        g
        for g in queue.items
        if g.status in ("pending", "active")
        and g.situation == queue.active_situation
    ]
    if not pending:
        pending = [g for g in queue.items if g.status in ("pending", "active")]
    if not pending:
        return None
    return sorted(pending, key=lambda x: -x.priority)[0]


def build_brief(
    queue: AdaptiveQueue | None = None,
    *,
    apply_best_transition: bool = False,
) -> AgentBrief:
    queue = queue or AdaptiveQueue.load()
    signals = collect_world_signals()
    queue = reconcile_queue(queue, signals)
    transitions = propose_transitions(signals, queue)

    if apply_best_transition and transitions:
        best = max(transitions, key=lambda t: t.confidence)
        if best.to_situation != queue.active_situation and best.confidence >= 0.8:
            queue.active_situation = best.to_situation
        if best.overlay is not None:
            queue.overlay = best.overlay
        elif best.overlay is None and best.reason.startswith("Real-data"):
            pass
        _log_transition(best, signals)

    state = queue.state()
    router = SituationRouter(state)
    goal = pick_active_goal(queue)
    cmds = GOAL_PLAYBOOK.get(goal.id, []) if goal else []
    if not cmds and transitions:
        t0 = transitions[0]
        if t0.to_situation == Situation.COLLECT_EPOC.value:
            cmds = GOAL_PLAYBOOK["goal-real-hand-60"][:2]

    narrative = _narrative_for(state, signals, goal)
    brief = AgentBrief(
        generated_at=time.time(),
        signals=signals,
        queue=queue,
        transitions=transitions,
        active_goal=goal,
        next_commands=cmds,
        hermes_block=router.hermes_context_block(),
        narrative=narrative,
    )
    queue.save()
    return brief


def _narrative_for(state: SituationState, signals: WorldSignals, goal: QueuedGoal | None) -> str:
    parts = [
        f"Situation `{state.effective().value}`: ",
    ]
    if signals.synthetic_only_data:
        parts.append("decoder runs on synthetic smoke — treat Unreal hooks as integration test only. ")
    if goal:
        parts.append(f"Focus queue goal `{goal.id}`. ")
    if signals.cortex_credentials:
        parts.append("Cortex path is configured. ")
    else:
        parts.append("No Cortex creds detected — collection blocked until .env set. ")
    return "".join(parts)


def _log_transition(t: TransitionProposal, signals: WorldSignals) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": time.time(),
        "type": "transition",
        "proposal": asdict(t),
        "signals": signals.to_dict(),
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def write_brief(path: Path = BRIEF_PATH, **kwargs: Any) -> Path:
    brief = build_brief(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(brief.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Adaptive engine — brief, reconcile, autopilot")
    p.add_argument("--brief", action="store_true", help="Write data/adaptive_brief.md")
    p.add_argument("--print-json", action="store_true")
    p.add_argument("--reconcile", action="store_true")
    p.add_argument("--autopilot", action="store_true", help="Apply strongest transition (>=0.8)")
    p.add_argument("--mark-pytest-ok", action="store_true")
    args = p.parse_args(argv)

    if args.mark_pytest_ok:
        (ROOT / "tests" / ".last_pytest_ok").write_text("1", encoding="utf-8")

    brief = build_brief(apply_best_transition=args.autopilot)
    if args.reconcile or args.brief or args.autopilot:
        brief.queue.save()

    if args.brief:
        pth = write_brief()
        print(f"wrote {pth}")

    if args.print_json or not any([args.brief, args.autopilot, args.mark_pytest_ok]):
        print(
            json.dumps(
                {
                    "signals": brief.signals.to_dict(),
                    "active_situation": brief.queue.active_situation,
                    "overlay": brief.queue.overlay,
                    "active_goal": brief.active_goal.id if brief.active_goal else None,
                    "next_commands": brief.next_commands,
                    "transitions": [asdict(t) for t in brief.transitions],
                    "hermes_block": brief.hermes_block,
                    "narrative": brief.narrative,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())