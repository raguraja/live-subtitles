# LiveSubtitles

Real-time English subtitle overlay for Windows 11. Captures system audio from any media player and transcribes it to English using Whisper AI.

![Windows 11](https://img.shields.io/badge/Windows-11-blue) ![Python 3.11](https://img.shields.io/badge/Python-3.11-blue) ![CUDA](https://img.shields.io/badge/CUDA-12.8-green)

## Features

- Captures audio from **any media player** (VLC, YouTube, Netflix, Spotify, etc.)
- Translates **any language** to English subtitles automatically
- Floating **always-on-top** overlay — stays above all windows
- **Click-through** — doesn't block mouse clicks on other apps
- GPU accelerated via CUDA (NVIDIA) for fast transcription
- Drag to reposition, right-click to quit

## Requirements

- Windows 11
- NVIDIA GPU (CUDA)
- Anaconda / Miniconda
- Python 3.11

## Installation

```bat
cd subtitle_app
install.bat
```

This creates a `subtitles` conda environment and installs all dependencies.

## Usage

### Quick start (double-click)
Double-click `run.bat`

### From terminal
```bat
conda activate subtitles
python subtitle_app.py
```

### Options
```
--model   tiny | base | small | medium | large-v2   (default: small)
--alpha   0.0 – 1.0 overlay transparency             (default: 0.85)
--device  force a specific loopback device index
```

Examples:
```bat
python subtitle_app.py --model medium        # better accuracy
python subtitle_app.py --alpha 0.6           # more transparent
python subtitle_app.py --device 14           # force device index
```

### Find your loopback device
```bat
conda activate subtitles
python check_devices.py
```

## Model comparison

| Model    | Latency | Accuracy | VRAM  |
|----------|---------|----------|-------|
| `tiny`   | ~0.3s   | OK       | ~0.5GB |
| `small`  | ~0.5s   | Good     | ~1GB  |
| `medium` | ~1s     | Better   | ~2.5GB |
| `large-v2` | ~2s  | Best     | ~5GB  |

## Controls

| Action | How |
|--------|-----|
| Move overlay | Left-click drag |
| Quit | Right-click → Quit |

## How it works

```
Any media player
    ↓ WASAPI loopback (system audio capture)
    ↓ Resample to 16kHz mono
    ↓ Energy-based VAD (speech detection)
    ↓ faster-whisper (CUDA) — translate to English
    ↓ Floating tkinter overlay
```
