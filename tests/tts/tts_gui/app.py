#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


DATA_ROOT = Path("/data")
BIN_ROOT = DATA_ROOT / "bin"
HOSTS_ROOT = DATA_ROOT / "etc" / "hosts"
PORTS_PATH = DATA_ROOT / "etc" / "ports.json"
TMP_ROOT = DATA_ROOT / "var" / "tmp" / "tts_gui"
RAM_TMP_ROOT = Path("/dev/shm/tts_gui")

ENGINE_OPTIONS = ("auto", "index", "fish", "cosy", "miotts", "piper")

ENGINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "auto": {},
    "index": {
        "no_speaker": False,
    },
    "fish": {
        "no_speaker": False,
        "prompt_text": "",
        "chunk_length": "",
        "temperature": "",
        "top_p": "",
        "repetition_penalty": "",
        "max_new_tokens": "",
        "max_length": "",
        "split_sentences": True,
        "chunk_chars": "",
        "speaker_seconds": "",
        "speaker_offset": "",
    },
    "cosy": {
        "no_speaker": False,
        "prompt_text": "",
    },
    "miotts": {
        "no_speaker": False,
    },
    "piper": {
        "speaker_id": "0",
        "length_scale": "",
        "noise_scale": "",
        "noise_w": "",
        "sentence_silence": "",
    },
}


def bootstrap_audio_env() -> None:
    if not os.environ.get("XDG_RUNTIME_DIR"):
        runtime_dir = f"/run/user/{os.getuid()}"
        if os.path.isdir(runtime_dir):
            os.environ["XDG_RUNTIME_DIR"] = runtime_dir
    if os.environ.get("XDG_RUNTIME_DIR") and not os.environ.get("PULSE_SERVER"):
        pulse_native = Path(os.environ["XDG_RUNTIME_DIR"]) / "pulse" / "native"
        if pulse_native.exists():
            os.environ["PULSE_SERVER"] = f"unix:{pulse_native}"


def host_short() -> str:
    host = os.environ.get("INFRA_HOST") or socket.gethostname()
    host = host.split(".", 1)[0].lower()
    if host.startswith("nitro"):
        return "nitro"
    if host.startswith("jetson"):
        return "jetson"
    return host


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def host_config() -> dict[str, Any]:
    short = host_short()
    for candidate in (HOSTS_ROOT / f"{short}.json", HOSTS_ROOT / f"{socket.gethostname()}.json"):
        if candidate.exists():
            return load_json(candidate)
    return {}


def tts_endpoint() -> tuple[str, int, str]:
    cfg = host_config()
    ports = load_json(PORTS_PATH)
    host = str(((cfg.get("tts") or {}).get("host")) or "127.0.0.1")
    base = int(((ports.get("services") or {}).get("tts") or {}).get("base") or 15101)
    instance = int(((cfg.get("tts") or {}).get("instance")) or 0)
    secret = str(((cfg.get("tts") or {}).get("secret_key")) or "")
    return host, base + instance, secret


def server_health(timeout: float = 1.5) -> dict[str, Any] | None:
    host, port, _secret = tts_endpoint()
    url = f"http://{host}:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def current_sink_name() -> str:
    bootstrap_audio_env()
    try:
        proc = subprocess.run(
            ["pactl", "info"],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except FileNotFoundError:
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    for line in proc.stdout.splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate() or 1
        frames = handle.getnframes()
    return float(frames) / float(rate)


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def list_piper_models() -> list[str]:
    root = DATA_ROOT / "models" / "piper"
    if not root.is_dir():
        return []
    return [str(p) for p in sorted(root.rglob("*.onnx"))]


def list_reference_voices() -> list[str]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key in seen or not path.is_file():
            return
        seen.add(key)
        candidates.append(path)

    fixed = [
        DATA_ROOT / "etc" / "tts" / "speaker.wav",
        DATA_ROOT / "tmp" / "speaker.wav",
        DATA_ROOT / "tmp" / "fish_ref.wav",
        DATA_ROOT / "var" / "tmp" / "tts_speaker_10s.wav",
    ]
    for p in fixed:
        add(p)

    voice_root = DATA_ROOT / "models" / "voices"
    if voice_root.is_dir():
        for p in sorted(voice_root.glob("*.wav")):
            add(p)

    return [str(p) for p in candidates]


def speaker_prompt_path(speaker_path: str) -> Path | None:
    if not speaker_path:
        return None
    path = Path(speaker_path)
    prompt_path = path.with_suffix(".txt")
    if prompt_path.is_file():
        return prompt_path
    return None


class TTSGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("TTS GUI")
        self.geometry("1240x860")
        self.minsize(1100, 760)

        if RAM_TMP_ROOT.parent.is_dir():
            RAM_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        TMP_ROOT.mkdir(parents=True, exist_ok=True)

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.ui_queue: queue.Queue[Any] = queue.Queue()
        self.busy = False

        self.engine_var = tk.StringVar(value="auto")
        self.mode_var = tk.StringVar(value="server")
        self.no_speaker_var = tk.BooleanVar(value=False)
        self.prompt_var = tk.StringVar()
        self.prompt_file_var = tk.StringVar(value="")
        self.chunk_length_var = tk.StringVar()
        self.temperature_var = tk.StringVar()
        self.top_p_var = tk.StringVar()
        self.repetition_penalty_var = tk.StringVar()
        self.max_new_tokens_var = tk.StringVar()
        self.max_length_var = tk.StringVar()
        self.split_sentences_var = tk.BooleanVar(value=True)
        self.chunk_chars_var = tk.StringVar()
        self.speaker_seconds_var = tk.StringVar()
        self.speaker_offset_var = tk.StringVar()
        self.piper_model_var = tk.StringVar()
        self.piper_speaker_id_var = tk.StringVar(value="0")
        self.piper_length_scale_var = tk.StringVar()
        self.piper_noise_scale_var = tk.StringVar()
        self.piper_noise_w_var = tk.StringVar()
        self.piper_sentence_silence_var = tk.StringVar()
        self.speaker_var = tk.StringVar()
        self.save_path_var = tk.StringVar(value="/data/tmp/tts_gui.wav")
        self.server_status_var = tk.StringVar(value="Server: checking...")
        self.server_detail_var = tk.StringVar(value="")
        self.sink_var = tk.StringVar(value=f"Sink: {current_sink_name()}")
        self.active_label_var = tk.StringVar(value="Idle")
        self.audio_status_var = tk.StringVar(value="Audio: none")
        self.playback_status_var = tk.StringVar(value="Playback: stopped")
        self.timeline_label_var = tk.StringVar(value="00:00 / 00:00")
        self.health_data: dict[str, Any] | None = None
        self.active_proc: subprocess.Popen[str] | None = None
        self.active_kind: str | None = None
        self.cancel_requested = False
        self.current_audio_path: Path | None = None
        self.current_audio_is_temporary = False
        self.current_audio_duration = 0.0
        self.play_proc: subprocess.Popen[str] | None = None
        self.play_started_at = 0.0
        self.play_offset = 0.0
        self.playback_after_id: str | None = None

        self._build_ui()
        self._refresh_voice_lists()
        self._apply_engine_defaults(self.engine_var.get().strip() or "auto")
        self._sync_option_visibility()
        self.engine_var.trace_add("write", lambda *_: self._on_engine_changed())
        self.no_speaker_var.trace_add("write", lambda *_: self._on_speaker_mode_changed())
        self.speaker_var.trace_add("write", lambda *_: self._on_speaker_changed())
        self.after(100, self._poll_log_queue)
        self.after(150, self._periodic_status_refresh)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, textvariable=self.server_status_var, font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(top, textvariable=self.server_detail_var).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.sink_var).grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(top, textvariable=self.active_label_var).grid(row=3, column=0, sticky="w", pady=(4, 0))

        btns = ttk.Frame(top)
        btns.grid(row=0, column=1, rowspan=4, sticky="e")
        ttk.Button(btns, text="Refresh", command=self.refresh_status).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="Start Server", command=self.start_server).grid(row=0, column=1, padx=4)
        ttk.Button(btns, text="Stop Server", command=self.stop_server).grid(row=0, column=2, padx=4)
        ttk.Button(btns, text="Stop Current", command=self.stop_current_action).grid(row=0, column=3, padx=4)

        self.progress = ttk.Progressbar(top, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        opts = ttk.LabelFrame(self, text="Options", padding=12)
        opts.grid(row=1, column=0, sticky="ew", padx=12)
        for col in range(6):
            opts.columnconfigure(col, weight=1 if col in (1, 3, 5) else 0)

        ttk.Label(opts, text="Mode").grid(row=0, column=0, sticky="w")
        mode_box = ttk.Frame(opts)
        mode_box.grid(row=0, column=1, sticky="ew")
        ttk.Radiobutton(mode_box, text="Server", variable=self.mode_var, value="server").pack(side="left")
        ttk.Radiobutton(mode_box, text="One-shot", variable=self.mode_var, value="oneshot").pack(side="left", padx=(8, 0))

        ttk.Label(opts, text="Engine").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.engine_combo = ttk.Combobox(opts, textvariable=self.engine_var, values=ENGINE_OPTIONS, state="readonly")
        self.engine_combo.grid(row=0, column=3, sticky="ew")

        self.speaker_label = ttk.Label(opts, text="Reference voice")
        self.speaker_label.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.speaker_combo = ttk.Combobox(opts, textvariable=self.speaker_var, state="readonly")
        self.speaker_combo.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.no_speaker_check = ttk.Checkbutton(opts, text="No speaker", variable=self.no_speaker_var)
        self.no_speaker_check.grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.refresh_voices_btn = ttk.Button(opts, text="Refresh Voices", command=self._refresh_voice_lists)
        self.refresh_voices_btn.grid(row=1, column=3, sticky="w", pady=(8, 0))

        self.piper_model_label = ttk.Label(opts, text="Piper model")
        self.piper_model_label.grid(row=1, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        self.piper_model_combo = ttk.Combobox(opts, textvariable=self.piper_model_var, state="readonly")
        self.piper_model_combo.grid(row=1, column=5, sticky="ew", pady=(8, 0))

        self.prompt_label = ttk.Label(opts, text="Prompt file")
        self.prompt_label.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.prompt_value_label = ttk.Label(opts, textvariable=self.prompt_file_var)
        self.prompt_value_label.grid(row=2, column=1, sticky="w", pady=(8, 0))
        self.chunk_label = ttk.Label(opts, text="Chunk length")
        self.chunk_label.grid(row=2, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.chunk_entry = ttk.Entry(opts, textvariable=self.chunk_length_var)
        self.chunk_entry.grid(row=2, column=3, sticky="ew", pady=(8, 0))

        self.temperature_label = ttk.Label(opts, text="Temperature")
        self.temperature_label.grid(row=2, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        self.temperature_entry = ttk.Entry(opts, textvariable=self.temperature_var)
        self.temperature_entry.grid(row=2, column=5, sticky="ew", pady=(8, 0))

        self.top_p_label = ttk.Label(opts, text="Top P")
        self.top_p_label.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.top_p_entry = ttk.Entry(opts, textvariable=self.top_p_var)
        self.top_p_entry.grid(row=3, column=1, sticky="ew", pady=(8, 0))
        self.repetition_penalty_label = ttk.Label(opts, text="Repetition penalty")
        self.repetition_penalty_label.grid(row=3, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.repetition_penalty_entry = ttk.Entry(opts, textvariable=self.repetition_penalty_var)
        self.repetition_penalty_entry.grid(row=3, column=3, sticky="ew", pady=(8, 0))
        self.max_new_tokens_label = ttk.Label(opts, text="Max new tokens")
        self.max_new_tokens_label.grid(row=3, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        self.max_new_tokens_entry = ttk.Entry(opts, textvariable=self.max_new_tokens_var)
        self.max_new_tokens_entry.grid(row=3, column=5, sticky="ew", pady=(8, 0))

        self.max_length_label = ttk.Label(opts, text="Max length")
        self.max_length_label.grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.max_length_entry = ttk.Entry(opts, textvariable=self.max_length_var)
        self.max_length_entry.grid(row=4, column=1, sticky="ew", pady=(8, 0))
        self.split_sentences_check = ttk.Checkbutton(opts, text="Split sentences", variable=self.split_sentences_var)
        self.split_sentences_check.grid(row=4, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.chunk_chars_label = ttk.Label(opts, text="Chunk chars")
        self.chunk_chars_label.grid(row=4, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        self.chunk_chars_entry = ttk.Entry(opts, textvariable=self.chunk_chars_var)
        self.chunk_chars_entry.grid(row=4, column=5, sticky="ew", pady=(8, 0))

        self.speaker_seconds_label = ttk.Label(opts, text="Speaker seconds")
        self.speaker_seconds_label.grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.speaker_seconds_entry = ttk.Entry(opts, textvariable=self.speaker_seconds_var)
        self.speaker_seconds_entry.grid(row=5, column=1, sticky="ew", pady=(8, 0))
        self.speaker_offset_label = ttk.Label(opts, text="Speaker offset")
        self.speaker_offset_label.grid(row=5, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.speaker_offset_entry = ttk.Entry(opts, textvariable=self.speaker_offset_var)
        self.speaker_offset_entry.grid(row=5, column=3, sticky="ew", pady=(8, 0))

        self.piper_speaker_id_label = ttk.Label(opts, text="Piper speaker-id")
        self.piper_speaker_id_label.grid(row=6, column=0, sticky="w", pady=(8, 0))
        self.piper_speaker_id_entry = ttk.Entry(opts, textvariable=self.piper_speaker_id_var)
        self.piper_speaker_id_entry.grid(row=6, column=1, sticky="ew", pady=(8, 0))
        self.piper_length_scale_label = ttk.Label(opts, text="Length scale")
        self.piper_length_scale_label.grid(row=6, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.piper_length_scale_entry = ttk.Entry(opts, textvariable=self.piper_length_scale_var)
        self.piper_length_scale_entry.grid(row=6, column=3, sticky="ew", pady=(8, 0))
        self.piper_noise_scale_label = ttk.Label(opts, text="Noise scale")
        self.piper_noise_scale_label.grid(row=6, column=4, sticky="w", padx=(12, 0), pady=(8, 0))
        self.piper_noise_scale_entry = ttk.Entry(opts, textvariable=self.piper_noise_scale_var)
        self.piper_noise_scale_entry.grid(row=6, column=5, sticky="ew", pady=(8, 0))

        self.piper_noise_w_label = ttk.Label(opts, text="Noise W")
        self.piper_noise_w_label.grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.piper_noise_w_entry = ttk.Entry(opts, textvariable=self.piper_noise_w_var)
        self.piper_noise_w_entry.grid(row=7, column=1, sticky="ew", pady=(8, 0))
        self.piper_sentence_silence_label = ttk.Label(opts, text="Sentence silence")
        self.piper_sentence_silence_label.grid(row=7, column=2, sticky="w", padx=(12, 0), pady=(8, 0))
        self.piper_sentence_silence_entry = ttk.Entry(opts, textvariable=self.piper_sentence_silence_var)
        self.piper_sentence_silence_entry.grid(row=7, column=3, sticky="ew", pady=(8, 0))

        body = ttk.Panedwindow(self, orient="vertical")
        body.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))

        text_frame = ttk.LabelFrame(body, text="Text", padding=8)
        log_frame = ttk.LabelFrame(body, text="Log", padding=8)
        body.add(text_frame, weight=4)
        body.add(log_frame, weight=1)

        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        self.text_box = ScrolledText(text_frame, wrap="word", font=("DejaVu Sans Mono", 12))
        self.text_box.grid(row=0, column=0, sticky="nsew")

        action_bar = ttk.Frame(text_frame)
        action_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(action_bar, text="Speak Using Selected Mode", command=self.speak_selected_mode).pack(side="left")
        ttk.Button(action_bar, text="Speak Once", command=lambda: self.speak("oneshot")).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Speak Via Server", command=lambda: self.speak("server")).pack(side="left", padx=(8, 0))
        ttk.Button(action_bar, text="Clear Text", command=lambda: self.text_box.delete("1.0", "end")).pack(side="right")

        player_frame = ttk.LabelFrame(text_frame, text="Player", padding=8)
        player_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        player_frame.columnconfigure(0, weight=1)

        ttk.Label(player_frame, textvariable=self.audio_status_var).grid(row=0, column=0, sticky="w")
        ttk.Label(player_frame, textvariable=self.playback_status_var).grid(row=0, column=1, sticky="e", padx=(12, 0))

        player_buttons = ttk.Frame(player_frame)
        player_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(player_buttons, text="Play", command=self.play_current_audio).pack(side="left")
        ttk.Button(player_buttons, text="Stop", command=self.stop_current_audio).pack(side="left", padx=(8, 0))
        ttk.Button(player_buttons, text="Save", command=self.save_current_audio).pack(side="left", padx=(8, 0))
        ttk.Label(player_buttons, textvariable=self.timeline_label_var).pack(side="right")

        self.timeline_canvas = tk.Canvas(player_frame, height=24, highlightthickness=1, highlightbackground="#808080")
        self.timeline_canvas.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.timeline_canvas.bind("<Configure>", lambda _event: self._draw_timeline())
        self.timeline_canvas.bind("<Button-1>", self._seek_timeline)

        save_row = ttk.Frame(player_frame)
        save_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        save_row.columnconfigure(1, weight=1)
        ttk.Label(save_row, text="Save as").grid(row=0, column=0, sticky="w")
        self.save_entry = ttk.Entry(save_row, textvariable=self.save_path_var)
        self.save_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_box = ScrolledText(log_frame, wrap="word", height=10, state="disabled")
        self.log_box.grid(row=0, column=0, sticky="nsew")

        self._sync_option_visibility()

    def _refresh_voice_lists(self) -> None:
        speakers = list_reference_voices()
        self.speaker_combo["values"] = speakers
        if speakers and self.speaker_var.get() not in speakers:
            self.speaker_var.set(speakers[0])

        piper_models = list_piper_models()
        self.piper_model_combo["values"] = piper_models
        if piper_models and self.piper_model_var.get() not in piper_models:
            preferred = next((p for p in piper_models if "de_DE-thorsten-high" in p), piper_models[0])
            self.piper_model_var.set(preferred)
        self._autofill_prompt_for_selected_speaker()

    def _log(self, message: str, level: str = "INFO") -> None:
        self.log_queue.put((level, message))

    def _ui_call(self, fn: Any) -> None:
        self.ui_queue.put(fn)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                ts = time.strftime("%H:%M:%S")
                self.log_box.insert("end", f"[{ts}] {level}: {message}\n")
                self.log_box.see("end")
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        try:
            while True:
                fn = self.ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.after(120, self._poll_log_queue)

    def _set_busy(self, value: bool) -> None:
        self.busy = value
        if value:
            self.progress.start(12)
        else:
            self.progress.stop()
            if self.play_proc is None or self.play_proc.poll() is not None:
                self.active_label_var.set("Idle")

    def _periodic_status_refresh(self) -> None:
        try:
            self.refresh_status()
        finally:
            self.after(3000, self._periodic_status_refresh)

    def _set_widget_group_visible(self, widgets: tuple[tk.Widget, ...], visible: bool) -> None:
        for widget in widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _apply_engine_defaults(self, engine: str) -> None:
        defaults = ENGINE_DEFAULTS.get(engine, {})
        if "no_speaker" in defaults:
            self.no_speaker_var.set(bool(defaults["no_speaker"]))
        self.prompt_var.set(str(defaults.get("prompt_text", "")))
        self.chunk_length_var.set(str(defaults.get("chunk_length", "")))
        self.temperature_var.set(str(defaults.get("temperature", "")))
        self.top_p_var.set(str(defaults.get("top_p", "")))
        self.repetition_penalty_var.set(str(defaults.get("repetition_penalty", "")))
        self.max_new_tokens_var.set(str(defaults.get("max_new_tokens", "")))
        self.max_length_var.set(str(defaults.get("max_length", "")))
        self.split_sentences_var.set(bool(defaults.get("split_sentences", True)))
        self.chunk_chars_var.set(str(defaults.get("chunk_chars", "")))
        self.speaker_seconds_var.set(str(defaults.get("speaker_seconds", "")))
        self.speaker_offset_var.set(str(defaults.get("speaker_offset", "")))
        self.piper_speaker_id_var.set(str(defaults.get("speaker_id", "0")))
        self.piper_length_scale_var.set(str(defaults.get("length_scale", "")))
        self.piper_noise_scale_var.set(str(defaults.get("noise_scale", "")))
        self.piper_noise_w_var.set(str(defaults.get("noise_w", "")))
        self.piper_sentence_silence_var.set(str(defaults.get("sentence_silence", "")))
        if engine == "piper" and self.piper_model_combo["values"] and not self.piper_model_var.get().strip():
            models = list(self.piper_model_combo["values"])
            preferred = next((p for p in models if "de_DE-thorsten-high" in p), models[0])
            self.piper_model_var.set(preferred)

    def _on_engine_changed(self) -> None:
        engine = self.engine_var.get().strip() or "auto"
        self._apply_engine_defaults(engine)
        self._sync_option_visibility()
        self._autofill_prompt_for_selected_speaker()

    def _on_speaker_mode_changed(self) -> None:
        self._sync_option_visibility()
        self._autofill_prompt_for_selected_speaker()

    def _on_speaker_changed(self) -> None:
        self._autofill_prompt_for_selected_speaker()

    def _autofill_prompt_for_selected_speaker(self) -> None:
        engine = self.engine_var.get().strip() or "auto"
        if engine not in {"fish", "cosy"}:
            self.prompt_var.set("")
            self.prompt_file_var.set("")
            return
        if self.no_speaker_var.get():
            self.prompt_var.set("")
            self.prompt_file_var.set("")
            return
        speaker = self.speaker_var.get().strip()
        prompt_path = speaker_prompt_path(speaker)
        if prompt_path is None:
            self.prompt_var.set("")
            self.prompt_file_var.set("")
            return
        try:
            self.prompt_var.set(prompt_path.read_text(encoding="utf-8").strip())
            self.prompt_file_var.set(prompt_path.name)
        except Exception:
            self.prompt_var.set("")
            self.prompt_file_var.set("")

    def _sync_option_visibility(self) -> None:
        engine = self.engine_var.get().strip() or "auto"
        show_ref = engine in {"index", "fish", "cosy", "miotts"}
        show_no_speaker = engine in {"fish", "cosy"}
        show_prompt = engine in {"fish", "cosy"}
        show_chunk = engine == "fish"
        show_fish_sampling = engine == "fish"
        show_fish_chunking = engine == "fish"
        show_fish_ref_window = engine == "fish"
        show_piper = engine == "piper"

        self._set_widget_group_visible(
            (self.speaker_label, self.speaker_combo, self.refresh_voices_btn),
            show_ref,
        )
        self._set_widget_group_visible((self.no_speaker_check,), show_no_speaker)
        self._set_widget_group_visible((self.prompt_label, self.prompt_value_label), show_prompt)
        self._set_widget_group_visible((self.chunk_label, self.chunk_entry), show_chunk)
        self._set_widget_group_visible((self.temperature_label, self.temperature_entry), show_fish_sampling)
        self._set_widget_group_visible((self.top_p_label, self.top_p_entry), show_fish_sampling)
        self._set_widget_group_visible((self.repetition_penalty_label, self.repetition_penalty_entry), show_fish_sampling)
        self._set_widget_group_visible((self.max_new_tokens_label, self.max_new_tokens_entry), show_fish_sampling)
        self._set_widget_group_visible((self.max_length_label, self.max_length_entry), show_fish_sampling)
        self._set_widget_group_visible((self.split_sentences_check,), show_fish_chunking)
        self._set_widget_group_visible((self.chunk_chars_label, self.chunk_chars_entry), show_fish_chunking)
        self._set_widget_group_visible((self.speaker_seconds_label, self.speaker_seconds_entry), show_fish_ref_window)
        self._set_widget_group_visible((self.speaker_offset_label, self.speaker_offset_entry), show_fish_ref_window)
        self._set_widget_group_visible((self.piper_model_label, self.piper_model_combo), show_piper)
        self._set_widget_group_visible(
            (
                self.piper_speaker_id_label,
                self.piper_speaker_id_entry,
                self.piper_length_scale_label,
                self.piper_length_scale_entry,
                self.piper_noise_scale_label,
                self.piper_noise_scale_entry,
                self.piper_noise_w_label,
                self.piper_noise_w_entry,
                self.piper_sentence_silence_label,
                self.piper_sentence_silence_entry,
            ),
            show_piper,
        )

    def _current_play_position(self) -> float:
        position = self.play_offset
        if self.play_proc is not None and self.play_proc.poll() is None:
            position += max(0.0, time.monotonic() - self.play_started_at)
        return min(self.current_audio_duration, position)

    def _set_audio_status(self, path: Path | None) -> None:
        if path is None:
            self.audio_status_var.set("Audio: none")
            self.timeline_label_var.set("00:00 / 00:00")
            return
        self.audio_status_var.set(f"Audio: {path.name} ({format_seconds(self.current_audio_duration)})")
        self.timeline_label_var.set(f"{format_seconds(self._current_play_position())} / {format_seconds(self.current_audio_duration)}")

    def _clear_playback_poll(self) -> None:
        if self.playback_after_id is not None:
            self.after_cancel(self.playback_after_id)
            self.playback_after_id = None

    def _draw_timeline(self) -> None:
        canvas = self.timeline_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.create_rectangle(0, 0, width, height, fill="#202020", outline="")
        if self.current_audio_duration > 0:
            progress = self._current_play_position() / self.current_audio_duration
            progress = min(1.0, max(0.0, progress))
            canvas.create_rectangle(0, 0, int(width * progress), height, fill="#4a90e2", outline="")
        canvas.create_rectangle(0, 0, width, height, outline="#808080")
        self.timeline_label_var.set(
            f"{format_seconds(self._current_play_position())} / {format_seconds(self.current_audio_duration)}"
        )

    def _delete_audio_file(self, path: Path | None, *, temporary: bool) -> None:
        if not temporary or path is None:
            return
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    def _reset_audio(self) -> None:
        self.stop_current_audio(reset_position=True)
        old_path = self.current_audio_path
        old_temporary = self.current_audio_is_temporary
        self.current_audio_path = None
        self.current_audio_is_temporary = False
        self.current_audio_duration = 0.0
        self.play_offset = 0.0
        self._delete_audio_file(old_path, temporary=old_temporary)
        self._set_audio_status(None)
        self.playback_status_var.set("Playback: stopped")
        self._draw_timeline()

    def _load_audio(self, path: Path, *, temporary: bool) -> None:
        self._reset_audio()
        self.current_audio_path = path
        self.current_audio_is_temporary = temporary
        self.current_audio_duration = wav_duration(path)
        self.play_offset = 0.0
        self.playback_status_var.set("Playback: ready")
        self._set_audio_status(path)
        self._draw_timeline()

    def _player_command(self, path: Path, offset: float) -> list[str]:
        bootstrap_audio_env()
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error"]
            if offset > 0.0:
                cmd.extend(["-ss", f"{offset:.3f}"])
            cmd.append(str(path))
            return cmd
        if offset <= 0.0:
            for name in ("paplay", "pw-play", "aplay"):
                if shutil.which(name):
                    return [name, str(path)]
        raise RuntimeError("no seekable audio playback command found")

    def _playback_finished(self) -> None:
        self._clear_playback_poll()
        self.play_proc = None
        self.play_started_at = 0.0
        self.play_offset = self.current_audio_duration
        if self.current_audio_path is not None:
            self.playback_status_var.set("Playback: finished")
            self._set_audio_status(self.current_audio_path)
        else:
            self.playback_status_var.set("Playback: stopped")
        if not self.busy:
            self.active_label_var.set("Idle")
        self._draw_timeline()

    def _poll_playback(self) -> None:
        if self.play_proc is None:
            self._clear_playback_poll()
            self._draw_timeline()
            return
        if self.play_proc.poll() is not None:
            self._playback_finished()
            return
        self._draw_timeline()
        self.playback_after_id = self.after(150, self._poll_playback)

    def _start_playback(self, offset: float) -> None:
        if self.current_audio_path is None:
            raise RuntimeError("no rendered audio available")
        self.stop_current_audio(reset_position=False)
        offset = min(self.current_audio_duration, max(0.0, offset))
        cmd = self._player_command(self.current_audio_path, offset)
        self.play_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=os.environ.copy(),
        )
        self.play_offset = offset
        self.play_started_at = time.monotonic()
        self.active_label_var.set("Playing audio")
        self.playback_status_var.set("Playback: playing")
        self._draw_timeline()
        self._poll_playback()

    def play_current_audio(self) -> None:
        if self.current_audio_path is None:
            self._log("No rendered audio available", "WARN")
            return
        try:
            offset = self._current_play_position()
            if offset >= self.current_audio_duration:
                offset = 0.0
            self._start_playback(offset)
        except Exception as exc:  # noqa: BLE001
            self._log(str(exc), "ERROR")

    def stop_current_audio(self, *, reset_position: bool = False) -> None:
        position = 0.0 if reset_position else self._current_play_position()
        self._clear_playback_poll()
        if self.play_proc is not None and self.play_proc.poll() is None:
            self.play_proc.terminate()
            try:
                self.play_proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.play_proc.kill()
                self.play_proc.wait(timeout=1.0)
        self.play_proc = None
        self.play_started_at = 0.0
        self.play_offset = 0.0 if reset_position else min(self.current_audio_duration, position)
        if self.current_audio_path is None:
            self.playback_status_var.set("Playback: stopped")
        elif reset_position:
            self.playback_status_var.set("Playback: stopped")
        else:
            self.playback_status_var.set("Playback: paused")
        if not self.busy:
            self.active_label_var.set("Idle" if self.play_proc is None else "Playing audio")
        if self.current_audio_path is not None:
            self._set_audio_status(self.current_audio_path)
        else:
            self._set_audio_status(None)
        self._draw_timeline()

    def _seek_timeline(self, event: Any) -> None:
        if self.current_audio_path is None or self.current_audio_duration <= 0:
            return
        width = max(1, self.timeline_canvas.winfo_width())
        offset = self.current_audio_duration * min(1.0, max(0.0, event.x / width))
        was_playing = self.play_proc is not None and self.play_proc.poll() is None
        self.stop_current_audio(reset_position=False)
        self.play_offset = offset
        self.playback_status_var.set("Playback: paused")
        self._set_audio_status(self.current_audio_path)
        self._draw_timeline()
        if was_playing:
            try:
                self._start_playback(offset)
            except Exception as exc:  # noqa: BLE001
                self._log(str(exc), "ERROR")

    def _stream_reader(self, pipe: Any, level: str) -> None:
        try:
            for line in iter(pipe.readline, ""):
                line = line.rstrip()
                if line:
                    self._log(line, level)
        finally:
            pipe.close()

    def _run_popen(
        self,
        cmd: list[str],
        *,
        label: str,
        kind: str,
        env: dict[str, str] | None = None,
        after_success: Any = None,
        refresh_after: bool = False,
        stop_server_on_cancel: bool = False,
        cleanup_path: Path | None = None,
        cleanup_on_error: bool = False,
    ) -> None:
        if self.busy:
            self._log("A TTS action is already running", "WARN")
            return
        self._set_busy(True)
        self.active_kind = kind
        self.active_label_var.set(label)
        self.cancel_requested = False

        def worker() -> None:
            proc: subprocess.Popen[str] | None = None
            try:
                self._log("CMD: " + " ".join(cmd))
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env or os.environ.copy(),
                )
                self.active_proc = proc
                if proc.stdout is not None:
                    threading.Thread(target=self._stream_reader, args=(proc.stdout, "INFO"), daemon=True).start()
                if proc.stderr is not None:
                    threading.Thread(target=self._stream_reader, args=(proc.stderr, "ERROR"), daemon=True).start()
                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"{label} failed with exit code {rc}")
                if after_success is not None:
                    after_success()
            except Exception as exc:  # noqa: BLE001
                if cleanup_on_error and cleanup_path is not None:
                    try:
                        cleanup_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                self._log(str(exc), "ERROR")
            finally:
                self.active_proc = None
                if stop_server_on_cancel and self.cancel_requested:
                    try:
                        proc2 = subprocess.run(
                            [str(BIN_ROOT / "stop_tts.sh")],
                            check=False,
                            capture_output=True,
                            text=True,
                            env=os.environ.copy(),
                        )
                        msg = ((proc2.stdout or "") + (proc2.stderr or "")).strip()
                        if msg:
                            self._log(msg)
                    except Exception as exc:  # noqa: BLE001
                        self._log(f"forced stop failed: {exc}", "ERROR")
                self.cancel_requested = False
                self.active_kind = None
                self._ui_call(lambda: self._set_busy(False))
                if refresh_after:
                    self._ui_call(self.refresh_status)

        threading.Thread(target=worker, daemon=True).start()

    def stop_current_action(self) -> None:
        if self.play_proc is not None and self.play_proc.poll() is None:
            self._log("Stopping current playback", "WARN")
            self.stop_current_audio(reset_position=False)
            return
        proc = self.active_proc
        if proc is None or proc.poll() is not None:
            self._log("No running action to stop", "WARN")
            return
        self.cancel_requested = True
        self._log(f"Stopping current action: {self.active_kind or 'process'}", "WARN")
        try:
            proc.terminate()
        except Exception as exc:  # noqa: BLE001
            self._log(f"terminate failed: {exc}", "ERROR")

    def refresh_status(self) -> None:
        self.sink_var.set(f"Sink: {current_sink_name()}")
        health = server_health()
        self.health_data = health
        if health:
            loaded = [k for k, v in (health.get("loaded") or {}).items() if v]
            resident = ", ".join(health.get("resident_loaded") or []) or "-"
            current_request = health.get("current_request")
            current_text = f" | Current: {current_request.get('engine')}" if isinstance(current_request, dict) and current_request.get("engine") else ""
            self.server_status_var.set(
                f"Server: running on {health.get('host')} pid={health.get('pid')} device={health.get('device')}"
            )
            self.server_detail_var.set(
                f"Loaded: {', '.join(loaded) or '-'} | Resident: {resident}{current_text}"
            )
            if not self.busy and isinstance(current_request, dict) and current_request.get("engine"):
                self.active_label_var.set(f"Server busy: {current_request.get('engine')}")
        else:
            host, port, _ = tts_endpoint()
            self.server_status_var.set(f"Server: not running on {host}:{port}")
            self.server_detail_var.set("")

    def start_server(self) -> None:
        env = os.environ.copy()
        selected_engine = self.engine_var.get().strip() or "auto"
        if selected_engine != "auto":
            env["TTS_PRELOAD"] = selected_engine
        else:
            env.pop("TTS_PRELOAD", None)
        self._run_popen(
            [str(BIN_ROOT / "tts.sh")],
            label="Starting server",
            kind="server_start",
            env=env,
            refresh_after=True,
            stop_server_on_cancel=True,
        )

    def stop_server(self) -> None:
        self._run_popen(
            [str(BIN_ROOT / "stop_tts.sh")],
            label="Stopping server",
            kind="server_stop",
            refresh_after=True,
        )

    def _text_value(self) -> str:
        return self.text_box.get("1.0", "end").strip()

    def _base_args(self) -> list[str]:
        args: list[str] = []
        engine = self.engine_var.get().strip() or "auto"
        args.extend(["--engine", engine])
        if engine in {"index", "miotts"}:
            speaker = self.speaker_var.get().strip()
            if speaker:
                args.extend(["--speaker", speaker])
        elif engine in {"fish", "cosy"}:
            if self.no_speaker_var.get():
                args.extend(["--speaker", "none"])
            else:
                speaker = self.speaker_var.get().strip()
                if speaker:
                    args.extend(["--speaker", speaker])
                    if self.prompt_var.get().strip():
                        args.extend(["--prompt-text", self.prompt_var.get().strip()])
            if engine == "fish":
                if self.chunk_length_var.get().strip():
                    args.extend(["--chunk-length", self.chunk_length_var.get().strip()])
                if self.temperature_var.get().strip():
                    args.extend(["--temperature", self.temperature_var.get().strip()])
                if self.top_p_var.get().strip():
                    args.extend(["--top-p", self.top_p_var.get().strip()])
                if self.repetition_penalty_var.get().strip():
                    args.extend(["--repetition-penalty", self.repetition_penalty_var.get().strip()])
                if self.max_new_tokens_var.get().strip():
                    args.extend(["--max-new-tokens", self.max_new_tokens_var.get().strip()])
                if self.max_length_var.get().strip():
                    args.extend(["--max-length", self.max_length_var.get().strip()])
                args.append("--split" if self.split_sentences_var.get() else "--no-split")
                if self.chunk_chars_var.get().strip():
                    args.extend(["--chunk-chars", self.chunk_chars_var.get().strip()])
                if self.speaker_seconds_var.get().strip():
                    args.extend(["--speaker-seconds", self.speaker_seconds_var.get().strip()])
                if self.speaker_offset_var.get().strip():
                    args.extend(["--speaker-offset", self.speaker_offset_var.get().strip()])
        elif engine == "piper":
            if self.piper_model_var.get().strip():
                args.extend(["--model", self.piper_model_var.get().strip()])
            if self.piper_speaker_id_var.get().strip():
                args.extend(["--speaker-id", self.piper_speaker_id_var.get().strip()])
            if self.piper_length_scale_var.get().strip():
                args.extend(["--length-scale", self.piper_length_scale_var.get().strip()])
            if self.piper_noise_scale_var.get().strip():
                args.extend(["--noise-scale", self.piper_noise_scale_var.get().strip()])
            if self.piper_noise_w_var.get().strip():
                args.extend(["--noise-w", self.piper_noise_w_var.get().strip()])
            if self.piper_sentence_silence_var.get().strip():
                args.extend(["--sentence-silence", self.piper_sentence_silence_var.get().strip()])
        return args

    def _output_target(self) -> tuple[Path, bool]:
        tmp_root = RAM_TMP_ROOT if RAM_TMP_ROOT.is_dir() else TMP_ROOT
        fd, tmp_name = tempfile.mkstemp(prefix="tts_gui_", suffix=".wav", dir=tmp_root)
        os.close(fd)
        return Path(tmp_name), True

    def save_current_audio(self) -> None:
        if self.current_audio_path is None or not self.current_audio_path.exists():
            self._log("No rendered audio available", "WARN")
            return
        try:
            target = Path(self.save_path_var.get().strip() or "/data/tmp/tts_gui.wav").expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.current_audio_path, target)
            self._log(f"Saved audio to {target}")
        except Exception as exc:  # noqa: BLE001
            self._log(str(exc), "ERROR")

    def speak_selected_mode(self) -> None:
        self.speak(self.mode_var.get())

    def speak(self, mode: str) -> None:
        text = self._text_value()
        if not text:
            messagebox.showerror("TTS GUI", "Text is empty.")
            return

        if mode == "server":
            self._speak_server(text)
        else:
            self._speak_oneshot(text)

    def _speak_oneshot(self, text: str) -> None:
        tmp_path, temporary = self._output_target()
        cmd = [str(BIN_ROOT / "tts_once.sh"), "-o", str(tmp_path), *self._base_args(), text]
        def after_success() -> None:
            self._ui_call(lambda path=tmp_path, keep=temporary: self._load_and_play_audio(path, keep))

        self._run_popen(
            cmd,
            label="Running one-shot TTS",
            kind="oneshot_speak",
            after_success=after_success,
            cleanup_path=tmp_path,
            cleanup_on_error=temporary,
        )

    def _speak_server(self, text: str) -> None:
        health = server_health()
        if not health:
            raise RuntimeError("TTS server is not running")

        tmp_path, temporary = self._output_target()
        cmd = [str(BIN_ROOT / "call_tts.sh"), "-o", str(tmp_path), *self._base_args(), text]

        def after_success() -> None:
            self._ui_call(lambda path=tmp_path, keep=temporary: self._load_and_play_audio(path, keep))

        self._run_popen(
            cmd,
            label="Running server TTS",
            kind="server_speak",
            after_success=after_success,
            refresh_after=True,
            stop_server_on_cancel=True,
            cleanup_path=tmp_path,
            cleanup_on_error=temporary,
        )

    def _load_and_play_audio(self, path: Path, temporary: bool) -> None:
        self._load_audio(path, temporary=temporary)
        try:
            self._start_playback(0.0)
        except Exception as exc:  # noqa: BLE001
            self.playback_status_var.set("Playback: ready")
            self._draw_timeline()
            self._log(str(exc), "ERROR")

    def _on_close(self) -> None:
        proc = self.active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        self._reset_audio()
        self.destroy()


def main() -> int:
    bootstrap_audio_env()
    app = TTSGui()
    app.mainloop()
    return 0
