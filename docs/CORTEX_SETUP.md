# EMOTIV Cortex setup (EPOC X + Premium)

## 1. Register an app

1. [Emotiv Developer](https://www.emotiv.com/developer/) → create Cortex app.
2. Copy **Client ID** and **Client Secret** into `.env` (never commit):

```bash
cp .env.example .env
# EMOTIV_CLIENT_ID=...
# EMOTIV_CLIENT_SECRET=...
```

3. First connect: approve the app in **EMOTIV Launcher** when prompted.

## 2. Premium raw EEG

Typing decoder needs **raw EEG** (`subscribe` stream `eeg`). Confirm your account has **Premium** access for the headset.

## 3. Run Cortex locally

- Install EMOTIV Launcher + connect EPOC X (good contact quality).
- Cortex WebSocket: `wss://localhost:6868` (default in client).

## 4. Record a session

```bash
pip install -e ".[recorder]"
python -m neuroglyph_recorder.session_runner --live --duration 120 --session ses-real-001
```

Outputs in `data/raw/`:
- `ses-real-001_meta.json` — key events
- `ses-real-001_eeg.csv` — 14-channel EEG

## 5. Preprocess + train

```bash
python -m neuroglyph_data.make_epochs_cli --raw data/raw --out data/processed --task hand \
  --clock-offset-ms 0
# If alignment is off:
python scripts/calibrate_clock.py --meta data/raw/ses-real-001_meta.json --offset-ms -120
```

## 6. Live demo loop

```bash
python scripts/live_bci_demo.py --live --checkpoint checkpoints/tiny_b2q_hand.pt
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `requestAccess` denied | Approve app in Launcher; check client id/secret |
| No headset | Wear headset; wait for `connected` in Launcher |
| Empty EEG CSV | Premium license; `subscribe` to `eeg` failed — check logs |
| Random val acc | Tune `clock_offset_ms`; more data; finetune per subject |

## API reference

[Connecting to Cortex API](https://emotiv.gitbook.io/cortex-api/connecting-to-the-cortex-api)