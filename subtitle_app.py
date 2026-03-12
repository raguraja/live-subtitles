#!/usr/bin/env python3
"""
LiveSubtitles — Real-time English subtitle overlay for Windows 11
Captures system audio (WASAPI loopback) and transcribes to English.

Usage:
    python subtitle_app.py [--model small|medium|large-v2] [--alpha 0.8]
"""

import argparse
import ctypes
import queue
import sys
import threading
import time
import logging
from collections import deque
from pathlib import Path

import numpy as np
import scipy.signal
import tkinter as tk
from tkinter import font as tkfont

# Log to file for debugging
logging.basicConfig(
    filename=str(Path(__file__).parent / "subtitle.log"),
    level=logging.DEBUG,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
    filemode="w",
)
log = logging.getLogger()

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_MODEL     = "small"
SAMPLE_RATE       = 16000
CAPTURE_RATE      = 48000
CHUNK_MS          = 100
SILENCE_MS        = 900       # ms of silence to end a segment
MAX_SEGMENT_SEC   = 12
MIN_SEGMENT_SEC   = 0.6
ENERGY_THRESHOLD  = 0.003     # RMS threshold to consider as speech
FONT_SIZE         = 22
BG_COLOR          = "#111111"
FG_COLOR          = "#ffffff"
OVERLAY_ALPHA     = 0.85
OVERLAY_WIDTH_PCT = 0.72
# ─────────────────────────────────────────────────────────────────────────────

segment_queue: queue.Queue = queue.Queue(maxsize=8)
result_queue:  queue.Queue = queue.Queue(maxsize=50)


# ── Audio capture ─────────────────────────────────────────────────────────────

def find_loopback_device():
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out_idx = wasapi_info["defaultOutputDevice"]
        default_out = p.get_device_info_by_index(default_out_idx)
        log.info(f"Default output device [{default_out_idx}]: {default_out['name']}")

        # Find loopback matching the default output
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice") and default_out["name"] in dev["name"]:
                log.info(f"Matched loopback [{i}]: {dev['name']}")
                return p, i, dev

        # Fallback: first available loopback
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice"):
                log.info(f"Fallback loopback [{i}]: {dev['name']}")
                return p, i, dev

    except Exception as e:
        log.error(f"Error finding loopback: {e}")
    return p, None, None


def resample_to_16k(audio: np.ndarray, src_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if src_rate != SAMPLE_RATE:
        gcd = np.gcd(SAMPLE_RATE, src_rate)
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE // gcd, src_rate // gcd)
    return audio.astype(np.float32)


class AudioCapture(threading.Thread):
    def __init__(self, device_idx, src_rate, channels):
        super().__init__(daemon=True, name="AudioCapture")
        self.device_idx = device_idx
        self.src_rate   = src_rate
        self.channels   = channels
        self.ring       = deque()
        self._stop      = threading.Event()

    def run(self):
        import pyaudiowpatch as pyaudio
        chunk_frames = int(self.src_rate * CHUNK_MS / 1000)
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.src_rate,
                input=True,
                input_device_index=self.device_idx,
                frames_per_buffer=chunk_frames,
            )
            log.info(f"Audio stream open: {self.src_rate}Hz {self.channels}ch")
            result_queue.put(("status", "Listening..."))
            while not self._stop.is_set():
                raw = stream.read(chunk_frames, exception_on_overflow=False)
                arr = np.frombuffer(raw, dtype=np.float32).copy()
                if self.channels > 1:
                    arr = arr.reshape(-1, self.channels)
                mono16k = resample_to_16k(arr, self.src_rate)
                self.ring.append(mono16k)
            stream.stop_stream()
            stream.close()
        except Exception as e:
            log.error(f"AudioCapture error: {e}")
            result_queue.put(("status", f"Audio error: {e}"))
        finally:
            p.terminate()

    def stop(self):
        self._stop.set()


# ── Energy-based VAD + Chunker ────────────────────────────────────────────────

class VADChunker(threading.Thread):
    """Simple energy-based voice activity detection — no downloads required."""

    def __init__(self, ring: deque):
        super().__init__(daemon=True, name="VADChunker")
        self.ring  = ring
        self._stop = threading.Event()

    def run(self):
        log.info("VADChunker started (energy-based VAD)")
        frame_size = int(SAMPLE_RATE * CHUNK_MS / 1000)   # ~1600 samples per 100ms
        silence_frames_needed = int(SILENCE_MS / CHUNK_MS)
        max_frames = int(MAX_SEGMENT_SEC * 1000 / CHUNK_MS)
        min_frames = int(MIN_SEGMENT_SEC * 1000 / CHUNK_MS)

        buffer     = []
        silent_cnt = 0
        in_speech  = False

        while not self._stop.is_set():
            if not self.ring:
                time.sleep(0.01)
                continue

            chunk = self.ring.popleft()
            rms   = float(np.sqrt(np.mean(chunk ** 2)))
            is_speech = rms > ENERGY_THRESHOLD

            if is_speech:
                buffer.append(chunk)
                silent_cnt = 0
                in_speech  = True
            elif in_speech:
                buffer.append(chunk)
                silent_cnt += 1
                if silent_cnt >= silence_frames_needed:
                    if len(buffer) >= min_frames:
                        self._flush(buffer)
                    buffer     = []
                    silent_cnt = 0
                    in_speech  = False

            if len(buffer) >= max_frames:
                if len(buffer) >= min_frames:
                    self._flush(buffer)
                buffer     = []
                silent_cnt = 0
                in_speech  = False

    def _flush(self, frames):
        audio = np.concatenate(frames)
        log.debug(f"Flushing segment: {len(audio)/SAMPLE_RATE:.1f}s")
        try:
            segment_queue.put_nowait(audio)
        except queue.Full:
            log.warning("Segment queue full, dropping segment")

    def stop(self):
        self._stop.set()


# ── Transcription ─────────────────────────────────────────────────────────────

class Transcriber(threading.Thread):
    def __init__(self, model_name: str):
        super().__init__(daemon=True, name="Transcriber")
        self.model_name = model_name
        self._stop      = threading.Event()

    def run(self):
        from faster_whisper import WhisperModel
        log.info(f"Loading Whisper model '{self.model_name}'...")
        result_queue.put(("status", f"Loading model '{self.model_name}'..."))
        try:
            model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
            log.info("Whisper loaded on CUDA")
        except Exception as e:
            log.warning(f"CUDA failed ({e}), using CPU")
            model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        result_queue.put(("status", "Ready — listening..."))

        while not self._stop.is_set():
            try:
                audio = segment_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                log.debug(f"Transcribing {len(audio)/SAMPLE_RATE:.1f}s segment")
                segments, info = model.transcribe(
                    audio,
                    task="translate",
                    language=None,
                    beam_size=5,
                    condition_on_previous_text=True,
                    vad_filter=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
                log.info(f"Transcribed [{info.language}]: {text[:80]}")
                if text:
                    lang = info.language if info.language != "en" else None
                    result_queue.put(("text", text, lang))
            except Exception as e:
                log.error(f"Transcription error: {e}")

    def stop(self):
        self._stop.set()


# ── Overlay UI ────────────────────────────────────────────────────────────────

class SubtitleOverlay:
    def __init__(self, alpha: float):
        self.root = tk.Tk()
        self.root.withdraw()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ow = int(sw * OVERLAY_WIDTH_PCT)

        ox = (sw - ow) // 2
        oy = sh - 130

        self.root.geometry(f"{ow}x110+{ox}+{oy}")
        self.root.overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-alpha", alpha)
        self.root.configure(bg=BG_COLOR)

        # Click-through
        try:
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            GWL_EXSTYLE = -20
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                                style | 0x80000 | 0x20)
        except Exception:
            pass

        f = tkfont.Font(family="Segoe UI", size=FONT_SIZE, weight="bold")
        self.label = tk.Label(
            self.root, text="LiveSubtitles starting...",
            font=f, fg=FG_COLOR, bg=BG_COLOR,
            wraplength=ow - 40, justify="center",
            padx=20, pady=12,
        )
        self.label.pack(fill="both", expand=True)

        self.lang_label = tk.Label(
            self.root, text="", fg="#999999", bg=BG_COLOR,
            font=tkfont.Font(family="Segoe UI", size=10),
        )
        self.lang_label.place(relx=1.0, rely=0.0, anchor="ne", x=-6, y=4)

        # Drag
        self._dx = self._dy = 0
        self.label.bind("<ButtonPress-1>", lambda e: setattr(self, '_dx', e.x_root - self.root.winfo_x()) or setattr(self, '_dy', e.y_root - self.root.winfo_y()))
        self.label.bind("<B1-Motion>",     lambda e: self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}"))

        # Right-click quit
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Quit", command=self.root.destroy)
        self.label.bind("<Button-3>", lambda e: menu.post(e.x_root, e.y_root))

        self._clear_job = None
        self.root.after(100, self._poll)
        self.root.deiconify()

    def _poll(self):
        try:
            while True:
                item = result_queue.get_nowait()
                if item[0] == "status":
                    self.label.config(text=item[1], fg="#aaaaaa")
                    self.lang_label.config(text="")
                elif item[0] == "text":
                    self.label.config(text=item[1], fg=FG_COLOR)
                    lang = item[2] if len(item) > 2 else None
                    self.lang_label.config(text=f"[{lang}→en]" if lang else "")
                    if self._clear_job:
                        self.root.after_cancel(self._clear_job)
                    self._clear_job = self.root.after(6000, lambda: (
                        self.label.config(text=""),
                        self.lang_label.config(text="")
                    ))
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def run(self):
        self.root.mainloop()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        choices=["tiny","base","small","medium","large-v2","large-v3"])
    parser.add_argument("--alpha",  type=float, default=OVERLAY_ALPHA)
    parser.add_argument("--device", type=int,   default=None,
                        help="Force loopback device index (see check_devices.py)")
    args = parser.parse_args()

    log.info(f"Starting LiveSubtitles, model={args.model}")

    p, dev_idx, dev_info = find_loopback_device()
    p.terminate()

    if args.device is not None:
        import pyaudiowpatch as pyaudio
        p2 = pyaudio.PyAudio()
        dev_info = p2.get_device_info_by_index(args.device)
        dev_idx  = args.device
        p2.terminate()
        log.info(f"Forced device [{dev_idx}]: {dev_info['name']}")

    if dev_idx is None:
        print("No WASAPI loopback device found.")
        sys.exit(1)

    src_rate = int(dev_info["defaultSampleRate"])
    channels = dev_info["maxInputChannels"]
    log.info(f"Capture: [{dev_idx}] {dev_info['name']} {src_rate}Hz {channels}ch")
    print(f"[Audio] {dev_info['name']} ({src_rate}Hz, {channels}ch)")

    capture     = AudioCapture(dev_idx, src_rate, channels)
    vad         = VADChunker(capture.ring)
    transcriber = Transcriber(args.model)

    capture.start()
    vad.start()
    transcriber.start()

    try:
        overlay = SubtitleOverlay(args.alpha)
        overlay.run()
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        vad.stop()
        transcriber.stop()
        log.info("Stopped.")


if __name__ == "__main__":
    main()
