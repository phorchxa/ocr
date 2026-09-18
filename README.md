# Finger Counter + Gesture Emoji

Real-time hand tracking from your webcam using MediaPipe's Hand Landmarker and OpenCV.
The script counts how many fingers are extended, recognizes common hand gestures, and
draws the matching emoji on the video feed.

## Requirements

- Python 3.12 (tested on macOS, Apple Silicon)
- A webcam
- The packages in `requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The hand landmark model (`hand_landmarker.task`) ships with the repo. If it is missing,
the script downloads it automatically from Google's MediaPipe model storage on first run.

## Usage

```bash
python gptst.py
```

| Key | Action |
| --- | --- |
| `q` or `Esc` | Quit |
| `s` | Save a screenshot as `finger_count_shot_NN.png` in the current directory |

## Recognized gestures

Gestures are matched on which fingers are extended. If the exact pattern is not in the
table, the closest pattern within one finger of difference is used, so slightly noisy
detections still resolve.

| Gesture | Emoji |
| --- | --- |
| Fist | 👊 |
| Thumbs up | 👍 |
| Pointing | ☝ |
| Peace | ✌ |
| Three | 🔢 |
| Four | 🖐 |
| Open palm | 🖐 |
| Call me | 🤙 |
| Rock on | 🤘 |
| Love | 🤟 |
| Pinky | 🤏 |
| Spider-man | 🕸 |
| Middle finger | 🖕 |

## How it works

1. OpenCV captures frames from the default camera.
2. MediaPipe Hand Landmarker returns 21 landmarks per detected hand.
3. Each finger is classified as extended or folded from landmark positions.
4. The five-finger pattern is looked up in the gesture table and the emoji is rendered
   with Pillow using the system emoji font, then composited back onto the frame.
