# First EPOC session (~10 min)

## Checklist

- Launcher + EPOC connected, good contacts
- Premium raw EEG
- `.env`: `EMOTIV_CLIENT_ID`, `EMOTIV_CLIENT_SECRET`
- `pip install -e ".[recorder]"`

## Type (English)

Cycle until time ends:

- **Left-heavy:** `fast left hand rest fast left hand rest`
- **Right-heavy:** `jump right hand pause jump right hand pause`
- **Balanced:** `the quick brown fox jumps over the lazy dog`

## Record

```bash
python scripts/collect_session.py --duration 600 --live
# first try: --duration 180
```

## Process + train (4070)

```bash
python scripts/test_gear_finetune.py    # PC pipeline test (no headset)
python scripts/cortex_probe.py          # mock EEG OK
python scripts/cortex_probe.py --live   # needs .env + Launcher + headset
python -m neuroglyph_data.make_epochs_cli --raw data/raw --out data/processed --task hand
python -m neuroglyph_train.train --task hand --epochs 40
python -m neuroglyph_train.evaluate --checkpoint checkpoints/tiny_b2q_hand.pt
```

**> 55%** hand val = first milestone; **> 60%** = live demo per queue goal.

## Realtime equipment

| Piece | Role |
|-------|------|
| EPOC X | Only sensor for v1 |
| PC + Cortex | `localhost:6868` |
| 4070 | Train only |
| Hermes/Unreal | After decode works in logs |