# Environments

## 1) Desktop (validation / benchmarking)

Recommended:
- Python `3.11`
- `venv` isolated environment
- ROI mode: `mediapipe` or `both`

Commands:

```bash
cd /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg
python3.11 -m venv .venv-desktop
source .venv-desktop/bin/activate
pip install -r requirements-desktop.txt
```

Run offline evaluation (both ROI modes):

```bash
python evaluate_dataset.py \
  --data-dir /Users/liangwenwang/Downloads/Code/demo1/data \
  --out-dir /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg/results \
  --roi-mode both
```

## 2) Raspberry Pi 5 (deployment)

Recommended:
- Python `3.11` (or distro default if stable)
- ROI mode default: `opencv` (lighter, more stable on RPi)
- Optional: test mediapipe only if performance is acceptable

Commands:

```bash
cd /Users/liangwenwang/Downloads/Code/demo1/rpi_rppg
python3 -m venv .venv-rpi
source .venv-rpi/bin/activate
pip install -r requirements-rpi.txt
```

Run realtime:

```bash
python main.py --source realsense --fps 60 --roi-mode opencv --show
```

## Notes

- If mediapipe import works but lacks `mp.solutions`, your package may be a `tasks`-only variant.
- In that case:
  - either keep `--roi-mode opencv`
  - or pass a tasks model path:

```bash
python main.py \
  --roi-mode mediapipe \
  --mp-face-detector-model /path/to/face_detection_short_range.tflite
```
