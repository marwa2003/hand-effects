# Hand Effects

Real-time hand-tracking visual effects with Python, OpenCV and MediaPipe.

The webcam follows both hands (21 landmarks each) and draws:

- a glowing skeleton with colors that shift over time
- pulsing points on every joint
- particles that burst from the fingertips
- a web of lines connecting the left and right hands

## Requirements

- Python 3.9 to 3.12 (MediaPipe does not support newer versions yet)
- A webcam

## Installation (Windows, PowerShell)

```powershell
python -m venv .venv310
.venv310\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python hanfcolors.py
```

Press `q` to quit.

## Customize

Edit the `CFG` dictionary at the top of `hanfcolors.py`:

| Setting | Effect |
|---|---|
| `hueSpeed` | speed of the color cycle |
| `lineWidth` / `nodeRadius` | thickness and point size |
| `particlesPerTip` | number of particles per fingertip |
| `smoothing` | tracking smoothness (0 to 0.9) |
| `glow` | neon glow on/off |

If the wrong camera opens, change `CAMERA_INDEX` in `hanfcolors.py`.

## Tech stack

Python, OpenCV, MediaPipe, NumPy
