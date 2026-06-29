# NeuroGlyph adaptive brief
_generated 2026-06-29 16:00:04_

## Situation
- **active:** `dev_pipeline`
- **overlay:** `None`
- **effective:** `dev_pipeline`

## World signals
```json
{
  "cortex_credentials": false,
  "raw_session_dirs": 0,
  "raw_eeg_rows_min": 0,
  "has_real_processed_hand": false,
  "synthetic_only_data": false,
  "checkpoint_exists": true,
  "checkpoint_val_acc": 1.0,
  "mindbot_live_steps": 0,
  "pytest_last_ok": true,
  "modal_cli": true
}
```

## Top transition proposals
- **Tests green, no hardware — dataset/Modal work is highest leverage** → `train_finetune` overlay=`None` (55%)

## Active queue goal
- **goal-real-hand-60** (P10): Hand classifier >60% validation accuracy on real EPOC typing sessions

## Suggested commands (agent: run highest feasible)

- `docs/CORTEX_SETUP.md — set EMOTIV_CLIENT_ID/SECRET in .env`
- `python scripts/collect_session.py --duration 600  # add --live when ready`
- `python -m neuroglyph_data.make_epochs_cli --raw data/raw --out data/processed --task hand`
- `python -m neuroglyph_train.finetune_subject --base-checkpoint checkpoints/tiny_b2q_hand.pt`
- `python -m neuroglyph_train.evaluate --checkpoint checkpoints/tiny_b2q_hand.pt`

## Narrative (MindBot / stream)

Situation `dev_pipeline`: Focus queue goal `goal-real-hand-60`. No Cortex creds detected — collection blocked until .env set. 
